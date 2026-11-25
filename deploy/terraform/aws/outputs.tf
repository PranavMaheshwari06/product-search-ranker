output "function_url" {
  value = aws_lambda_function_url.search.function_url
}

output "bucket_name" {
  value = aws_s3_bucket.data.bucket
}

output "ecr_repository_url" {
  value = aws_ecr_repository.lambda.repository_url
}
