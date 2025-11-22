resource "aws_dynamodb_table" "model_registry" {
  name           = "${var.project_name}-model-registry-${var.environment}"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "model_version"
  range_key      = "timestamp"

  attribute {
    name = "model_version"
    type = "S"
  }

  attribute {
    name = "timestamp"
    type = "S"
  }

  attribute {
    name = "status"
    type = "S"
  }

  global_secondary_index {
    name            = "status-index"
    hash_key        = "status"
    range_key       = "timestamp"
    projection_type = "ALL"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  tags = var.tags
}

resource "aws_dynamodb_table" "experiments" {
  name           = "${var.project_name}-experiments-${var.environment}"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "experiment_id"
  range_key      = "run_id"

  attribute {
    name = "experiment_id"
    type = "S"
  }

  attribute {
    name = "run_id"
    type = "S"
  }

  tags = var.tags
}

resource "aws_dynamodb_table" "inference_logs" {
  name           = "${var.project_name}-inference-logs-${var.environment}"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "log_id"
  range_key      = "timestamp"

  attribute {
    name = "log_id"
    type = "S"
  }

  attribute {
    name = "timestamp"
    type = "S"
  }

  attribute {
    name = "user_id"
    type = "S"
  }

  global_secondary_index {
    name            = "user-index"
    hash_key        = "user_id"
    range_key       = "timestamp"
    projection_type = "ALL"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  stream_enabled   = true
  stream_view_type = "NEW_AND_OLD_IMAGES"

  tags = var.tags
}
