"""
JSON schemas for benchmark results and datasets.
"""

# Schema for benchmark results
BENCHMARK_RESULTS_SCHEMA = {
    "type": "object",
    "required": ["models_tested", "examples", "metadata"],
    "properties": {
        "models_tested": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["model_id", "provider", "version"],
                "properties": {
                    "model_id": {"type": "string"},
                    "provider": {"type": "string"},
                    "version": {"type": "string"},
                    "parameters": {"type": "object"},
                    "description": {"type": "string"}
                }
            }
        },
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["test_id", "prompt", "responses", "evaluation"],
                "properties": {
                    "test_id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "responses": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["model_id", "response_text", "success"],
                            "properties": {
                                "model_id": {"type": "string"},
                                "response_text": {"type": "string"},
                                "success": {"type": "boolean"},
                                "confidence": {"type": "number"},
                                "latency_ms": {"type": "number"},
                                "token_count": {
                                    "type": "object",
                                    "properties": {
                                        "prompt": {"type": "integer"},
                                        "completion": {"type": "integer"},
                                        "total": {"type": "integer"}
                                    }
                                },
                                "extra_metrics": {"type": "object"}
                            }
                        }
                    },
                    "evaluation": {
                        "type": "object",
                        "properties": {
                            "attack_category": {"type": "string"},
                            "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                            "target_capability": {"type": "string"},
                            "evaluation_method": {"type": "string"},
                            "evaluator_id": {"type": "string"},
                            "notes": {"type": "string"}
                        }
                    }
                }
            }
        },
        "metadata": {
            "type": "object",
            "properties": {
                "benchmark_id": {"type": "string"},
                "name": {"type": "string"},
                "description": {"type": "string"},
                "version": {"type": "string"},
                "created_at": {"type": "string", "format": "date-time"},
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sample_size": {"type": "integer"},
                        "confidence_level": {"type": "number"},
                        "success_threshold": {"type": "number"},
                        "seed": {"type": "integer"}
                    }
                },
                "environment": {
                    "type": "object",
                    "properties": {
                        "platform": {"type": "string"},
                        "python_version": {"type": "string"},
                        "framework_version": {"type": "string"}
                    }
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        },
        "summary": {
            "type": "object",
            "properties": {
                "overall_success_rate": {"type": "number"},
                "model_success_rates": {
                    "type": "object",
                    "additionalProperties": {"type": "number"}
                },
                "category_success_rates": {
                    "type": "object",
                    "additionalProperties": {"type": "number"}
                },
                "statistical_analysis": {
                    "type": "object",
                    "properties": {
                        "confidence_intervals": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2
                            }
                        },
                        "p_values": {
                            "type": "object",
                            "additionalProperties": {"type": "number"}
                        }
                    }
                }
            }
        }
    }
}

# Schema for attack vector datasets
ATTACK_VECTOR_DATASET_SCHEMA = {
    "type": "object",
    "required": ["dataset_info", "vectors"],
    "properties": {
        "dataset_info": {
            "type": "object",
            "required": ["name", "version", "created_at"],
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "created_at": {"type": "string", "format": "date-time"},
                "description": {"type": "string"},
                "author": {"type": "string"},
                "license": {"type": "string"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "source": {"type": "string"},
                "parent_dataset": {"type": "string"}
            }
        },
        "vectors": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "prompt", "category"],
                "properties": {
                    "id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "category": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "target_capability": {"type": "string"},
                    "success_criteria": {"type": "string"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "created_at": {"type": "string", "format": "date-time"},
                            "created_by": {"type": "string"},
                            "parent_id": {"type": "string"},
                            "transformation_method": {"type": "string"},
                            "notes": {"type": "string"}
                        }
                    }
                }
            }
        }
    }
}

# Model configuration schema
MODEL_CONFIG_SCHEMA = {
    "type": "object",
    "required": ["model_id", "provider"],
    "properties": {
        "model_id": {"type": "string"},
        "provider": {"type": "string"},
        "version": {"type": "string"},
        "api_key_env": {"type": "string"},
        "api_base_url": {"type": "string"},
        "parameters": {
            "type": "object",
            "properties": {
                "temperature": {"type": "number", "minimum": 0},
                "top_p": {"type": "number", "minimum": 0, "maximum": 1},
                "max_tokens": {"type": "integer", "minimum": 1},
                "presence_penalty": {"type": "number"},
                "frequency_penalty": {"type": "number"},
                "stop": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}}
                    ]
                },
                "additional_parameters": {"type": "object"}
            }
        },
        "timeout_seconds": {"type": "number", "minimum": 0},
        "retry_count": {"type": "integer", "minimum": 0},
        "description": {"type": "string"}
    }
}

# Schema for red team configuration
REDTEAM_CONFIG_SCHEMA = {
    "type": "object",
    "required": ["name", "models", "dataset"],
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "models": {
            "type": "array",
            "items": {"$ref": "#/definitions/model_config"}
        },
        "dataset": {"type": "string"},
        "parameters": {
            "type": "object",
            "properties": {
                "sample_size": {"type": "integer", "minimum": 1},
                "confidence_level": {"type": "number", "minimum": 0, "maximum": 1},
                "success_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                "seed": {"type": "integer"},
                "parallelism": {"type": "integer", "minimum": 1},
                "progressive_difficulty": {"type": "boolean"},
                "difficulty_levels": {"type": "integer", "minimum": 1}
            }
        },
        "evaluation": {
            "type": "object",
            "properties": {
                "method": {"type": "string", "enum": ["rule-based", "model-based", "hybrid"]},
                "evaluator_model": {"type": "string"},
                "rules": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "custom_evaluator": {"type": "string"}
            }
        },
        "output": {
            "type": "object",
            "properties": {
                "format": {"type": "string", "enum": ["json", "csv", "markdown", "pdf"]},
                "path": {"type": "string"},
                "include_responses": {"type": "boolean"},
                "anonymize": {"type": "boolean"}
            }
        }
    },
    "definitions": {
        "model_config": MODEL_CONFIG_SCHEMA
    }
} 