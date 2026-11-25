terraform {
  required_version = ">= 1.9"

  required_providers {
    aws = {
      source = "hashicorp/aws"
      # 6.28 is the first release where aws_lambda_function_url also adds lambda:InvokeFunction
      # for a public URL; earlier 6.x adds only lambda:InvokeFunctionUrl.
      version = "~> 6.28"
    }
  }
}
