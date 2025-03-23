"""
Kubernetes job manager for distributed red teaming workloads.
"""

import os
import json
import time
import logging
import base64
import uuid
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import tempfile

# Import Kubernetes client libraries conditionally - they will only be required if using K8s functionality
try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False


class K8sJobManager:
    """
    Kubernetes job manager for distributed red teaming workloads.
    
    Handles creation, monitoring, and cleanup of distributed red teaming jobs
    across a Kubernetes cluster.
    """
    
    def __init__(
        self, 
        namespace: str = "default", 
        image: str = "redteamer:latest",
        job_ttl_seconds: int = 3600,
        service_account: Optional[str] = None,
        config_file: Optional[str] = None,
        context: Optional[str] = None,
        in_cluster: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the Kubernetes job manager.
        
        Args:
            namespace: Kubernetes namespace to use for jobs
            image: Docker image to use for red teaming jobs
            job_ttl_seconds: Time-to-live for completed jobs in seconds
            service_account: Service account to use for jobs
            config_file: Path to kubeconfig file (if not using in-cluster config)
            context: Kubernetes context to use
            in_cluster: Whether to use in-cluster configuration
            verbose: Whether to log detailed information
        """
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        
        if not KUBERNETES_AVAILABLE:
            self.logger.error("Kubernetes client libraries not available. Install with `pip install kubernetes`")
            raise ImportError("Kubernetes client libraries not available. Install with `pip install kubernetes`")
        
        self.namespace = namespace
        self.image = image
        self.job_ttl_seconds = job_ttl_seconds
        self.service_account = service_account
        self.verbose = verbose
        
        # Initialize Kubernetes client
        try:
            if in_cluster:
                config.load_incluster_config()
                self.logger.debug("Using in-cluster Kubernetes configuration")
            else:
                config.load_kube_config(config_file=config_file, context=context)
                self.logger.debug(f"Using kubeconfig: {config_file or 'default'}, context: {context or 'default'}")
            
            self.batch_api = client.BatchV1Api()
            self.core_api = client.CoreV1Api()
            self.api_client = client.ApiClient()
            
            self.logger.info(f"Kubernetes client initialized, using namespace {namespace}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise
        
        # Track created jobs
        self.jobs = {}
    
    def create_config_map(self, name: str, data: Dict[str, str]) -> str:
        """
        Create a ConfigMap in the Kubernetes cluster.
        
        Args:
            name: Name of the ConfigMap
            data: Dictionary of key-value pairs to store in the ConfigMap
            
        Returns:
            Name of the created ConfigMap
        """
        try:
            # Add a unique suffix to prevent conflicts
            unique_suffix = str(uuid.uuid4())[:8]
            full_name = f"{name}-{unique_suffix}"
            
            # Create ConfigMap object
            config_map = client.V1ConfigMap(
                api_version="v1",
                kind="ConfigMap",
                metadata=client.V1ObjectMeta(name=full_name),
                data=data
            )
            
            # Create ConfigMap in the cluster
            self.core_api.create_namespaced_config_map(
                namespace=self.namespace,
                body=config_map
            )
            
            self.logger.debug(f"Created ConfigMap {full_name} in namespace {self.namespace}")
            return full_name
        except ApiException as e:
            self.logger.error(f"Failed to create ConfigMap: {e}")
            raise
    
    def _create_job_spec(
        self, 
        job_name: str, 
        config_map_name: str,
        config_path: str,
        parallelism: int = 1,
        completions: int = 1,
        backoff_limit: int = 3,
        active_deadline_seconds: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None
    ) -> client.V1Job:
        """
        Create a Kubernetes Job specification.
        
        Args:
            job_name: Name of the job
            config_map_name: Name of the ConfigMap containing the red team config
            config_path: Path in the container where the config should be mounted
            parallelism: Number of parallel pods to run
            completions: Number of successful completions required
            backoff_limit: Number of retries before considering the job failed
            active_deadline_seconds: Deadline for job completion in seconds
            env_vars: Environment variables to set in the container
            
        Returns:
            Kubernetes Job specification
        """
        # Set up environment variables
        env = []
        if env_vars:
            for key, value in env_vars.items():
                env.append(client.V1EnvVar(name=key, value=value))
        
        # Create volume mount for the config map
        volume_mounts = [
            client.V1VolumeMount(
                name="config-volume",
                mount_path=os.path.dirname(config_path),
                read_only=True
            )
        ]
        
        # Create volume for the config map
        volumes = [
            client.V1Volume(
                name="config-volume",
                config_map=client.V1ConfigMapVolumeSource(
                    name=config_map_name,
                    items=[
                        client.V1KeyToPath(
                            key="config.json",
                            path=os.path.basename(config_path)
                        )
                    ]
                )
            )
        ]
        
        # Create container spec
        container = client.V1Container(
            name="redteamer",
            image=self.image,
            command=["redteamer", "run", "--config", config_path, "--non-interactive"],
            volume_mounts=volume_mounts,
            env=env,
            resources=client.V1ResourceRequirements(
                requests={"cpu": "500m", "memory": "1Gi"},
                limits={"cpu": "2", "memory": "4Gi"}
            )
        )
        
        # Create pod template
        pod_template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={"job-name": job_name}),
            spec=client.V1PodSpec(
                containers=[container],
                volumes=volumes,
                restart_policy="Never",
                service_account_name=self.service_account
            )
        )
        
        # Create job spec
        job_spec = client.V1JobSpec(
            parallelism=parallelism,
            completions=completions,
            backoff_limit=backoff_limit,
            template=pod_template,
            ttl_seconds_after_finished=self.job_ttl_seconds
        )
        
        if active_deadline_seconds:
            job_spec.active_deadline_seconds = active_deadline_seconds
        
        # Create job
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(name=job_name),
            spec=job_spec
        )
        
        return job
    
    def launch_redteam_job(
        self, 
        config: Dict,
        job_name: Optional[str] = None,
        parallelism: int = 4,
        active_deadline_seconds: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Launch a red team evaluation as a Kubernetes job.
        
        Args:
            config: Red team configuration
            job_name: Name for the job (auto-generated if not provided)
            parallelism: Number of parallel workers to use
            active_deadline_seconds: Maximum time for job execution
            env_vars: Environment variables to pass to the job
            
        Returns:
            Job ID (name) of the created job
        """
        try:
            # Generate a job name if not provided
            if not job_name:
                job_name = f"redteam-{str(uuid.uuid4())[:8]}"
            
            # Ensure job name is valid for Kubernetes
            job_name = job_name.lower().replace("_", "-")
            
            # Create a ConfigMap with the red team configuration
            config_data = {"config.json": json.dumps(config, indent=2)}
            config_map_name = self.create_config_map(f"redteam-config-{job_name}", config_data)
            
            # Path in the container where config will be mounted
            config_path = "/etc/redteamer/config.json"
            
            # Create the job spec
            job = self._create_job_spec(
                job_name=job_name,
                config_map_name=config_map_name,
                config_path=config_path,
                parallelism=parallelism,
                completions=1,  # We only need one successful completion
                backoff_limit=3,
                active_deadline_seconds=active_deadline_seconds,
                env_vars=env_vars
            )
            
            # Create the job
            response = self.batch_api.create_namespaced_job(
                namespace=self.namespace,
                body=job
            )
            
            # Store job details
            self.jobs[job_name] = {
                "name": job_name,
                "config_map": config_map_name,
                "status": "created",
                "created_at": time.time(),
                "config": config
            }
            
            self.logger.info(f"Created job {job_name} in namespace {self.namespace}")
            if self.verbose:
                self.logger.debug(f"Job spec: {job}")
            
            return job_name
        
        except ApiException as e:
            self.logger.error(f"Failed to create job: {e}")
            raise
    
    def get_job_status(self, job_name: str) -> Dict:
        """
        Get the status of a red team job.
        
        Args:
            job_name: Name of the job
            
        Returns:
            Dictionary with job status information
        """
        try:
            # Get the job from Kubernetes API
            job = self.batch_api.read_namespaced_job(
                name=job_name,
                namespace=self.namespace
            )
            
            # Extract status information
            status = {
                "name": job_name,
                "active": job.status.active or 0,
                "succeeded": job.status.succeeded or 0,
                "failed": job.status.failed or 0,
                "creation_time": job.metadata.creation_timestamp,
                "completion_time": job.status.completion_time,
                "conditions": []
            }
            
            # Add conditions if available
            if job.status.conditions:
                for condition in job.status.conditions:
                    status["conditions"].append({
                        "type": condition.type,
                        "status": condition.status,
                        "reason": condition.reason,
                        "message": condition.message,
                        "last_transition_time": condition.last_transition_time
                    })
            
            # Add status from our tracking
            if job_name in self.jobs:
                status.update({
                    "config_map": self.jobs[job_name].get("config_map"),
                    "created_at": self.jobs[job_name].get("created_at")
                })
            
            # Determine overall status
            if status["succeeded"] > 0:
                status["status"] = "succeeded"
            elif status["failed"] > 0:
                status["status"] = "failed"
            elif status["active"] > 0:
                status["status"] = "running"
            else:
                status["status"] = "pending"
            
            return status
        
        except ApiException as e:
            if e.status == 404:
                self.logger.warning(f"Job {job_name} not found in namespace {self.namespace}")
                return {"name": job_name, "status": "not_found"}
            else:
                self.logger.error(f"Failed to get job status: {e}")
                raise
    
    def get_job_logs(self, job_name: str) -> List[str]:
        """
        Get logs from a red team job.
        
        Args:
            job_name: Name of the job
            
        Returns:
            List of log lines from the job pods
        """
        try:
            # Get pods belonging to the job
            pod_list = self.core_api.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"job-name={job_name}"
            )
            
            logs = []
            for pod in pod_list.items:
                pod_name = pod.metadata.name
                try:
                    pod_logs = self.core_api.read_namespaced_pod_log(
                        name=pod_name,
                        namespace=self.namespace
                    )
                    logs.append(f"--- Logs from pod {pod_name} ---")
                    logs.append(pod_logs)
                except ApiException as e:
                    logs.append(f"Error getting logs from pod {pod_name}: {e}")
            
            return logs
        
        except ApiException as e:
            self.logger.error(f"Failed to get job logs: {e}")
            raise
    
    def get_job_results(self, job_name: str) -> Optional[Dict]:
        """
        Get results from a completed red team job.
        
        Args:
            job_name: Name of the job
            
        Returns:
            Dictionary with red team results or None if not available
        """
        # Check job status first
        status = self.get_job_status(job_name)
        if status["status"] != "succeeded":
            self.logger.warning(f"Job {job_name} has not completed successfully. Status: {status['status']}")
            return None
        
        # For now, we'll return the logs which should contain the results
        # In a more sophisticated implementation, results would be saved to a PVC or object storage
        logs = self.get_job_logs(job_name)
        
        # Try to parse JSON results from logs
        results = None
        for log in logs:
            try:
                # Look for JSON output in the logs
                json_start = log.find('{"results":')
                if json_start >= 0:
                    json_text = log[json_start:]
                    results = json.loads(json_text)
                    break
            except json.JSONDecodeError:
                continue
        
        return results
    
    def delete_job(self, job_name: str, delete_config_map: bool = True) -> bool:
        """
        Delete a red team job and optionally its associated ConfigMap.
        
        Args:
            job_name: Name of the job to delete
            delete_config_map: Whether to also delete the associated ConfigMap
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get config map name before deleting job
            config_map_name = None
            if job_name in self.jobs:
                config_map_name = self.jobs[job_name].get("config_map")
            
            # Delete the job
            self.batch_api.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                body=client.V1DeleteOptions(
                    propagation_policy="Background"
                )
            )
            
            self.logger.info(f"Deleted job {job_name} from namespace {self.namespace}")
            
            # Delete the config map if requested
            if delete_config_map and config_map_name:
                try:
                    self.core_api.delete_namespaced_config_map(
                        name=config_map_name,
                        namespace=self.namespace
                    )
                    self.logger.info(f"Deleted config map {config_map_name} from namespace {self.namespace}")
                except ApiException as e:
                    self.logger.warning(f"Failed to delete config map {config_map_name}: {e}")
            
            # Remove from our tracking
            if job_name in self.jobs:
                del self.jobs[job_name]
            
            return True
        
        except ApiException as e:
            if e.status == 404:
                self.logger.warning(f"Job {job_name} not found in namespace {self.namespace}")
                return False
            else:
                self.logger.error(f"Failed to delete job: {e}")
                raise
    
    def list_jobs(self) -> List[Dict]:
        """
        List all red team jobs in the namespace.
        
        Returns:
            List of job status dictionaries
        """
        try:
            # Get all jobs with a redteam label
            job_list = self.batch_api.list_namespaced_job(
                namespace=self.namespace,
                label_selector="app=redteamer"
            )
            
            jobs = []
            for job in job_list.items:
                job_name = job.metadata.name
                jobs.append(self.get_job_status(job_name))
            
            return jobs
        
        except ApiException as e:
            self.logger.error(f"Failed to list jobs: {e}")
            raise
    
    def wait_for_job_completion(self, job_name: str, timeout_seconds: int = 600, poll_interval: int = 10) -> Dict:
        """
        Wait for a job to complete.
        
        Args:
            job_name: Name of the job to wait for
            timeout_seconds: Maximum time to wait in seconds
            poll_interval: How often to check job status in seconds
            
        Returns:
            Final job status
        """
        start_time = time.time()
        elapsed = 0
        
        while elapsed < timeout_seconds:
            status = self.get_job_status(job_name)
            
            if status["status"] in ["succeeded", "failed"]:
                return status
            
            # Sleep before next check
            time.sleep(poll_interval)
            elapsed = time.time() - start_time
        
        # If we get here, we've timed out
        self.logger.warning(f"Timeout waiting for job {job_name} to complete")
        return self.get_job_status(job_name) 