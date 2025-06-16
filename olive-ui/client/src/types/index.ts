// Type definitions for Olive UI

export interface Pass {
  name: string;
  module_path: string;
  supported_accelerators: string[];
  supported_precisions: string[];
  extra_dependencies: string[];
  dataset_required: string;
  category: string;
  subcategory: string;
}

export interface PassWithFramework extends Pass {
  framework: string;
}

export interface PassesByCategory {
  [category: string]: {
    [subcategory: string]: Pass[];
  };
}

export interface SelectedPass {
  id: string;
  name: string;
  type: string;
  config: Record<string, any>;
  passInfo: Pass;
}

export interface InputModel {
  type: string;
  model_path: string;
  task?: string;
  [key: string]: any;
}

export interface WorkflowConfig {
  input_model: InputModel;
  passes: Record<string, {
    type: string;
    config: Record<string, any>;
  }>;
  output_dir: string;
  cli_command?: string;
}

export interface ModelOutput {
  model_id: string;
  model_path: string;
  model_type: string;
  from_pass?: string;
  device?: string;
  execution_provider?: string;
  parent_model_id?: string;
  metrics?: Record<string, any>;
}

export interface WorkflowOutput {
  available_devices?: string[];
  best_model?: ModelOutput & {
    metrics?: Record<string, any>;
    metrics_value?: Record<string, any>;
  };
  output_models?: ModelOutput[];
  run_history?: Record<string, any>;
  [key: string]: any;
}

export interface Job {
  job_id: string;
  status: 'running' | 'completed' | 'failed';
  output: string | null;
  error: string | null;
  workflow_output?: WorkflowOutput;
  created_at?: string;
  completed_at?: string;
  config?: WorkflowConfig;
}

export interface CLICommand {
  name: string;
  description: string;
  args: Record<string, CLIArgument>;
}

export interface CLIArgument {
  type: 'string' | 'boolean' | 'file' | 'choice';
  required?: boolean;
  default?: any;
  description: string;
  choices?: string[];
}

export interface PassSchema {
  [key: string]: {
    type: string;
    default?: any;
    required?: boolean;
    description?: string;
    choices?: any[];
  };
}

export interface PassSchemaResponse {
  pass_name: string;
  schema: PassSchema;
  pass_info: {
    module_path: string;
    supported_accelerators: string[];
    supported_precisions: string[];
    extra_dependencies: string[];
    category: string;
    subcategory: string;
    dataset_required: string;
  };
}