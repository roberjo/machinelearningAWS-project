variable "project_name" {
  type = string
}
variable "environment" {
  type = string
}
variable "container_image" {
  type = string
}
variable "model_bucket" {
  type = string
}
variable "model_version" {
  type = string
}
variable "data_bucket" {
  type = string
}
variable "tags" {
  type = map(string)
  default = {}
}
