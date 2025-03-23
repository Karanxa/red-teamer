"""
Dataset management system for LLM red teaming.
"""

import os
import json
import uuid
import datetime
import hashlib
from typing import Dict, List, Optional, Union, Any, Set
import logging
from pathlib import Path

import jsonschema
import pandas as pd

from redteamer.config.schemas import ATTACK_VECTOR_DATASET_SCHEMA

class DatasetManager:
    """
    Manages attack vector datasets for LLM red teaming.
    
    Features:
    - Dataset loading, validation, and versioning
    - Attack vector categorization and metadata management
    - Dataset augmentation and quality control
    - Efficient storage and retrieval
    """
    
    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize a DatasetManager.
        
        Args:
            dataset_path: Path to an existing dataset file.
        """
        self.logger = logging.getLogger(__name__)
        self.dataset = {
            "dataset_info": {
                "name": "",
                "version": "0.1.0",
                "created_at": datetime.datetime.now().isoformat(),
                "description": "",
                "author": "",
                "license": "",
                "tags": [],
                "source": "",
                "parent_dataset": None
            },
            "vectors": []
        }
        
        if dataset_path:
            self.load_dataset(dataset_path)
    
    def load_dataset(self, path: str) -> Dict:
        """
        Load and validate a dataset from a file.
        
        Args:
            path: Path to the dataset file.
            
        Returns:
            The loaded dataset.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            # Validate against schema
            jsonschema.validate(instance=dataset, schema=ATTACK_VECTOR_DATASET_SCHEMA)
            self.dataset = dataset
            self.logger.info(f"Loaded dataset {dataset['dataset_info']['name']} with {len(dataset['vectors'])} vectors")
            return dataset
        except (json.JSONDecodeError, jsonschema.exceptions.ValidationError) as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise ValueError(f"Invalid dataset format: {e}")
    
    def save_dataset(self, path: str) -> None:
        """
        Save the dataset to a file.
        
        Args:
            path: Path to save the dataset.
        """
        try:
            # Validate before saving
            jsonschema.validate(instance=self.dataset, schema=ATTACK_VECTOR_DATASET_SCHEMA)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.dataset, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved dataset to {path}")
        except (IOError, jsonschema.exceptions.ValidationError) as e:
            self.logger.error(f"Error saving dataset: {e}")
            raise
    
    def create_dataset(self, name: str, description: str = "", author: str = "", 
                      license: str = "", tags: List[str] = None) -> None:
        """
        Create a new empty dataset.
        
        Args:
            name: Name of the dataset.
            description: Description of the dataset.
            author: Author of the dataset.
            license: License for the dataset.
            tags: Tags for the dataset.
        """
        self.dataset = {
            "dataset_info": {
                "name": name,
                "version": "0.1.0",
                "created_at": datetime.datetime.now().isoformat(),
                "description": description,
                "author": author,
                "license": license,
                "tags": tags or [],
                "source": "",
                "parent_dataset": None
            },
            "vectors": []
        }
        self.logger.info(f"Created new dataset: {name}")
    
    def get_vector_by_id(self, vector_id: str) -> Optional[Dict]:
        """
        Get a vector by its ID.
        
        Args:
            vector_id: ID of the vector to get.
            
        Returns:
            The vector if found, None otherwise.
        """
        for vector in self.dataset["vectors"]:
            if vector["id"] == vector_id:
                return vector
        return None
    
    def add_vector(self, prompt: str, category: str, severity: str = "medium",
                  target_capability: str = "", success_criteria: str = "",
                  tags: List[str] = None, metadata: Dict = None) -> str:
        """
        Add a new attack vector to the dataset.
        
        Args:
            prompt: The prompt text.
            category: Category of the attack vector.
            severity: Severity level (low, medium, high, critical).
            target_capability: The target capability being tested.
            success_criteria: Criteria for determining success.
            tags: Tags for the vector.
            metadata: Additional metadata.
            
        Returns:
            ID of the new vector.
        """
        # Create a unique ID based on content hash and timestamp
        content_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()[:8]
        timestamp = int(datetime.datetime.now().timestamp())
        vector_id = f"v_{content_hash}_{timestamp}"
        
        vector = {
            "id": vector_id,
            "prompt": prompt,
            "category": category,
            "severity": severity,
            "target_capability": target_capability,
            "success_criteria": success_criteria,
            "tags": tags or [],
            "metadata": metadata or {
                "created_at": datetime.datetime.now().isoformat(),
            }
        }
        
        self.dataset["vectors"].append(vector)
        self.logger.info(f"Added vector {vector_id} to dataset")
        return vector_id
    
    def remove_vector(self, vector_id: str) -> bool:
        """
        Remove a vector from the dataset.
        
        Args:
            vector_id: ID of the vector to remove.
            
        Returns:
            True if the vector was removed, False otherwise.
        """
        for i, vector in enumerate(self.dataset["vectors"]):
            if vector["id"] == vector_id:
                self.dataset["vectors"].pop(i)
                self.logger.info(f"Removed vector {vector_id} from dataset")
                return True
        return False
    
    def update_vector(self, vector_id: str, **updates) -> bool:
        """
        Update a vector in the dataset.
        
        Args:
            vector_id: ID of the vector to update.
            **updates: Fields to update.
            
        Returns:
            True if the vector was updated, False otherwise.
        """
        for vector in self.dataset["vectors"]:
            if vector["id"] == vector_id:
                for key, value in updates.items():
                    if key in vector:
                        vector[key] = value
                self.logger.info(f"Updated vector {vector_id}")
                return True
        return False
    
    def filter_vectors(self, **filters) -> List[Dict]:
        """
        Filter vectors by specified criteria.
        
        Args:
            **filters: Criteria to filter by.
            
        Returns:
            List of vectors that match the criteria.
        """
        results = []
        
        for vector in self.dataset["vectors"]:
            match = True
            
            for key, value in filters.items():
                if key not in vector or vector[key] != value:
                    match = False
                    break
            
            if match:
                results.append(vector)
        
        return results
    
    def get_categories(self) -> List[str]:
        """
        Get all unique categories in the dataset.
        
        Returns:
            List of unique categories.
        """
        return list(set(v["category"] for v in self.dataset["vectors"]))
    
    def get_tags(self) -> List[str]:
        """
        Get all unique tags in the dataset.
        
        Returns:
            List of unique tags.
        """
        tags = set()
        for vector in self.dataset["vectors"]:
            tags.update(vector.get("tags", []))
        return list(tags)
    
    def create_version(self, new_version: str, description: str = "") -> Dict:
        """
        Create a new version of the dataset.
        
        Args:
            new_version: Version string for the new version.
            description: Description of the changes.
            
        Returns:
            The new dataset.
        """
        new_dataset = {
            "dataset_info": {
                **self.dataset["dataset_info"],
                "version": new_version,
                "created_at": datetime.datetime.now().isoformat(),
                "description": description,
                "parent_dataset": f"{self.dataset['dataset_info']['name']}:{self.dataset['dataset_info']['version']}"
            },
            "vectors": self.dataset["vectors"].copy()
        }
        
        self.dataset = new_dataset
        self.logger.info(f"Created new dataset version: {new_version}")
        return new_dataset
    
    def deduplicate(self) -> int:
        """
        Remove duplicate vectors from the dataset.
        
        Returns:
            Number of duplicates removed.
        """
        seen_hashes = set()
        unique_vectors = []
        duplicates = 0
        
        for vector in self.dataset["vectors"]:
            # Create a hash of the prompt to identify duplicates
            prompt_hash = hashlib.md5(vector["prompt"].encode('utf-8')).hexdigest()
            
            if prompt_hash not in seen_hashes:
                seen_hashes.add(prompt_hash)
                unique_vectors.append(vector)
            else:
                duplicates += 1
        
        self.dataset["vectors"] = unique_vectors
        self.logger.info(f"Removed {duplicates} duplicate vectors")
        return duplicates
    
    def augment(self, technique: str, source_ids: List[str] = None, count: int = 10) -> List[str]:
        """
        Augment the dataset with new vectors derived from existing ones.
        
        Args:
            technique: Augmentation technique to use.
            source_ids: IDs of vectors to augment. If None, all vectors are used.
            count: Number of new vectors to generate.
            
        Returns:
            List of IDs of the new vectors.
        """
        # This is a placeholder. In a real implementation, you would use 
        # NLP techniques to transform prompts.
        
        if technique not in ["paraphrase", "translate", "synonym_replacement", "back_translation"]:
            raise ValueError(f"Unknown augmentation technique: {technique}")
        
        source_vectors = []
        if source_ids:
            for vector_id in source_ids:
                vector = self.get_vector_by_id(vector_id)
                if vector:
                    source_vectors.append(vector)
        else:
            source_vectors = self.dataset["vectors"]
        
        if not source_vectors:
            return []
        
        new_ids = []
        for _ in range(count):
            # In a real implementation, this would use NLP to transform the prompt
            source = source_vectors[_ % len(source_vectors)]
            new_prompt = f"Augmented ({technique}): {source['prompt']}"
            
            new_id = self.add_vector(
                prompt=new_prompt,
                category=source["category"],
                severity=source["severity"],
                target_capability=source["target_capability"],
                success_criteria=source["success_criteria"],
                tags=source.get("tags", []) + [f"augmented_{technique}"],
                metadata={
                    "source": "augmentation",
                    "created_at": datetime.datetime.now().isoformat(),
                    "parent_id": source["id"],
                    "transformation_method": technique,
                }
            )
            new_ids.append(new_id)
        
        self.logger.info(f"Created {len(new_ids)} new vectors using {technique}")
        return new_ids
    
    def get_dataset(self) -> Dict:
        """
        Get the complete dataset.
        
        Returns:
            The dataset dictionary.
        """
        return self.dataset
    
    def get_vectors(self) -> List[Dict]:
        """
        Get all vectors in the dataset.
        
        Returns:
            List of vectors.
        """
        return self.dataset["vectors"]
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary of statistics.
        """
        vectors = self.dataset["vectors"]
        categories = {}
        severities = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        targets = {}
        tags_count = {}
        
        for vector in vectors:
            # Count categories
            category = vector["category"]
            categories[category] = categories.get(category, 0) + 1
            
            # Count severities
            severity = vector.get("severity", "medium")
            severities[severity] = severities.get(severity, 0) + 1
            
            # Count target capabilities
            target = vector.get("target_capability", "")
            if target:
                targets[target] = targets.get(target, 0) + 1
            
            # Count tags
            for tag in vector.get("tags", []):
                tags_count[tag] = tags_count.get(tag, 0) + 1
        
        return {
            "total_vectors": len(vectors),
            "categories": categories,
            "severities": severities,
            "targets": targets,
            "tags": tags_count
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the dataset to a pandas DataFrame.
        
        Returns:
            DataFrame representation of the dataset.
        """
        return pd.DataFrame(self.dataset["vectors"])
    
    def from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Update the dataset from a pandas DataFrame.
        
        Args:
            df: DataFrame containing vector data.
        """
        self.dataset["vectors"] = df.to_dict(orient="records")
        self.logger.info(f"Updated dataset from DataFrame with {len(df)} rows")
    
    def merge_datasets(self, other_dataset: Union[Dict, 'DatasetManager']) -> None:
        """
        Merge another dataset into this one.
        
        Args:
            other_dataset: Another DatasetManager or dataset dictionary.
        """
        if isinstance(other_dataset, DatasetManager):
            other_dict = other_dataset.get_dataset()
        else:
            other_dict = other_dataset
        
        # Add vectors from other dataset, avoiding duplicates
        existing_ids = {v["id"] for v in self.dataset["vectors"]}
        for vector in other_dict["vectors"]:
            if vector["id"] not in existing_ids:
                self.dataset["vectors"].append(vector)
                existing_ids.add(vector["id"])
        
        # Update dataset info
        self.dataset["dataset_info"]["version"] = f"{self.dataset['dataset_info']['version']}+{other_dict['dataset_info']['version']}"
        
        # Update tags
        self.dataset["dataset_info"]["tags"] = list(set(
            self.dataset["dataset_info"]["tags"] + other_dict["dataset_info"]["tags"]
        ))
        
        self.logger.info(f"Merged dataset {other_dict['dataset_info']['name']} into current dataset") 