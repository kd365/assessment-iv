output "cluster_name" {
  value = data.aws_eks_cluster.cluster.name
}

output "cluster_endpoint" {
  value = data.aws_eks_cluster.cluster.endpoint
}

output "namespaces" {
  value = [
    kubernetes_namespace.client_a.metadata[0].name,
    kubernetes_namespace.client_b.metadata[0].name,
    kubernetes_namespace.client_c.metadata[0].name,
  ]
}
