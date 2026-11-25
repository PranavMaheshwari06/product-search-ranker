variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "name" {
  description = "Prefix for resource names."
  type        = string
  default     = "psr"
}

variable "image_uri" {
  description = "ECR image for the Lambda function; the image is built and pushed after the repository exists."
  type        = string
}

variable "budget_limit_usd" {
  description = "Monthly cost budget in USD."
  type        = number
}

variable "budget_email" {
  description = "Address that receives the budget alerts."
  type        = string
}

variable "reserved_concurrency" {
  description = "Max concurrent Lambda runs. A new account with a 10-run quota rejects any reservation; set -1 there."
  type        = number
  default     = 5
}
