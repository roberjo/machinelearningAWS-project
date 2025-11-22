resource "aws_sagemaker_model" "recommender" {
  name               = "${var.project_name}-model-${var.environment}"
  execution_role_arn = aws_iam_role.sagemaker_role.arn

  primary_container {
    image          = var.container_image
    model_data_url = "s3://${var.model_bucket}/${var.model_version}/model.tar.gz"
    environment = {
      MODEL_VERSION = var.model_version
    }
  }

  tags = var.tags
}

resource "aws_sagemaker_endpoint_configuration" "recommender" {
  name = "${var.project_name}-endpoint-config-${var.environment}-${var.model_version}"

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.recommender.name
    initial_instance_count = 0  # Serverless

    serverless_config {
      max_concurrency   = 20
      memory_size_in_mb = 4096
    }
  }

  tags = var.tags
}

resource "aws_sagemaker_endpoint" "recommender" {
  name                 = "${var.project_name}-endpoint-${var.environment}"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.recommender.name

  tags = var.tags
}

# IAM Role for SageMaker
resource "aws_iam_role" "sagemaker_role" {
  name = "${var.project_name}-sagemaker-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "sagemaker.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy" "sagemaker_s3_access" {
  name = "s3-access"
  role = aws_iam_role.sagemaker_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.model_bucket}/*",
          "arn:aws:s3:::${var.data_bucket}/*"
        ]
      }
    ]
  })
}
