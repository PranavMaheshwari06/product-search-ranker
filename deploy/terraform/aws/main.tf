provider "aws" {
  region = var.region

  default_tags {
    tags = {
      project = "product-search-ranker"
    }
  }
}

resource "aws_s3_bucket" "data" {
  bucket_prefix = "${var.name}-"
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Daily publishes keep every old version; expire them so storage stays inside the budget.
resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "expire-old-versions"
    status = "Enabled"
    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_ecr_repository" "lambda" {
  name = var.name

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Storage is billed per GB; only the newest images are ever deployed.
resource "aws_ecr_lifecycle_policy" "lambda" {
  repository = aws_ecr_repository.lambda.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep the last 3 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 3
      }
      action = { type = "expire" }
    }]
  })
}

resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${var.name}"
  retention_in_days = 7
}

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "lambda" {
  statement {
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.lambda.arn}:*"]
  }

  statement {
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.data.arn}/*"]
  }

  statement {
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn]
  }
}

resource "aws_iam_role" "lambda" {
  name               = "${var.name}-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy" "lambda" {
  name   = "${var.name}-lambda"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.lambda.json
}

resource "aws_lambda_function" "search" {
  function_name = var.name
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = var.image_uri
  memory_size   = 3008
  timeout       = 30
  # Caps what a public URL can be made to cost; -1 removes the cap (see the variable).
  reserved_concurrent_executions = var.reserved_concurrency

  environment {
    variables = {
      PSR_BUCKET = aws_s3_bucket.data.bucket
    }
  }

  # Lambda would otherwise create the log group itself, without the retention.
  depends_on = [aws_cloudwatch_log_group.lambda]
}

# With authorization_type NONE the provider also adds the lambda:InvokeFunctionUrl and
# lambda:InvokeFunction (InvokedViaFunctionUrl) permissions for principal "*", so no
# aws_lambda_permission is declared here.
resource "aws_lambda_function_url" "search" {
  function_name      = aws_lambda_function.search.function_name
  authorization_type = "NONE"

  cors {
    allow_origins = ["*"]
    allow_methods = ["GET"]
  }
}

resource "aws_budgets_budget" "monthly" {
  name         = "${var.name}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.budget_limit_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_email]
  }
}
