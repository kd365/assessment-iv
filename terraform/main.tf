# Namespaces
resource "kubernetes_namespace" "client_a" {
  metadata {
    name = "client-a"
    labels = { team = "client-a" }
  }
}

resource "kubernetes_namespace" "client_b" {
  metadata {
    name = "client-b"
    labels = { team = "client-b" }
  }
}

resource "kubernetes_namespace" "client_c" {
  metadata {
    name = "client-c"
    labels = { team = "client-c" }
  }
}

# ConfigMaps
resource "kubernetes_config_map" "client_a" {
  metadata {
    name      = "client-a-config"
    namespace = kubernetes_namespace.client_a.metadata[0].name
  }
  data = {
    ENDPOINT_NAME      = "credit-xgboost-endpoint"
    AWS_DEFAULT_REGION = "us-west-2"
    LOG_LEVEL          = "INFO"
    MODEL_TYPE         = "xgboost"
    TEAM_NAME          = "client-a-financial-services"
  }
}

resource "kubernetes_config_map" "client_b" {
  metadata {
    name      = "client-b-config"
    namespace = kubernetes_namespace.client_b.metadata[0].name
  }
  data = {
    ENDPOINT_NAME      = "park-clustering-kmeans-endpoint"
    AWS_DEFAULT_REGION = "us-west-2"
    LOG_LEVEL          = "INFO"
    MODEL_TYPE         = "kmeans"
    TEAM_NAME          = "client-b-outdoor-recreation"
  }
}

resource "kubernetes_config_map" "client_c" {
  metadata {
    name      = "client-c-config"
    namespace = kubernetes_namespace.client_c.metadata[0].name
  }
  data = {
    ENDPOINT_NAME      = "legal-nlp-ner-endpoint"
    AWS_DEFAULT_REGION = "us-west-2"
    LOG_LEVEL          = "INFO"
    MODEL_TYPE         = "huggingface-ner"
    TEAM_NAME          = "client-c-legal-tech"
  }
}

# Resource Quotas
resource "kubernetes_resource_quota" "client_a" {
  metadata {
    name      = "client-a-quota"
    namespace = kubernetes_namespace.client_a.metadata[0].name
  }
  spec {
    hard = {
      "requests.cpu"    = "1"
      "limits.cpu"      = "2"
      "requests.memory" = "2Gi"
      "limits.memory"   = "4Gi"
      pods              = "5"
    }
  }
}

resource "kubernetes_resource_quota" "client_b" {
  metadata {
    name      = "client-b-quota"
    namespace = kubernetes_namespace.client_b.metadata[0].name
  }
  spec {
    hard = {
      "requests.cpu"    = "1"
      "limits.cpu"      = "2"
      "requests.memory" = "2Gi"
      "limits.memory"   = "4Gi"
      pods              = "5"
    }
  }
}

resource "kubernetes_resource_quota" "client_c" {
  metadata {
    name      = "client-c-quota"
    namespace = kubernetes_namespace.client_c.metadata[0].name
  }
  spec {
    hard = {
      "requests.cpu"    = "1"
      "limits.cpu"      = "2"
      "requests.memory" = "2Gi"
      "limits.memory"   = "4Gi"
      pods              = "5"
    }
  }
}
