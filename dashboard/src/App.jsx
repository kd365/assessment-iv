import { useState, useEffect } from 'react'

const DEMO_RESPONSES = {
  'Credit Risk API': {
    predict: { prediction: 0.4819, confidence: 'medium', model_version: 'v1', endpoint_used: 'credit-xgboost-endpoint' },
  },
  'Park Clustering API': {
    predict: { cluster: 3, distance: 1.7689 },
  },
  'Legal NLP API': {
    predict: {
      entities: [
        { word: 'Acme Corp', entity: 'B-ORG', score: 0.999, start: 0, end: 9 },
        { word: 'John Smith', entity: 'B-PER', score: 0.998, start: 31, end: 41 },
        { word: 'December 31, 2025', entity: 'B-DATE', score: 0.97, start: 45, end: 62 },
      ],
      text: 'Acme Corp shall pay $50,000 to John Smith by December 31, 2025.',
    },
  },
}

const SERVICES = [
  {
    name: 'Credit Risk API',
    team: 'Client A - Financial Services',
    model: 'XGBoost (Binary Classification)',
    endpoint: 'credit-xgboost-endpoint',
    namespace: 'client-a',
    url: import.meta.env.VITE_CLIENT_A_URL || 'http://k8s-clienta-creditri-9ea6d388a9-aee104298a79ef50.elb.us-east-1.amazonaws.com',
    featureCount: 43,
    testPayload: Array(43).fill(0.5),
  },
  {
    name: 'Park Clustering API',
    team: 'Client B - Outdoor Recreation',
    model: 'K-Means (Clustering)',
    endpoint: 'park-clustering-kmeans-endpoint',
    namespace: 'client-b',
    url: import.meta.env.VITE_CLIENT_B_URL || 'http://k8s-clientb-parkclus-1366fbc92c-e8e34ea80d5e7570.elb.us-east-1.amazonaws.com',
    featureCount: 12,
    testPayload: [0.5, 0.5, 1, 1, 0, 1, 0, 1, 0, 0, 0.4, 0.6],
  },
  {
    name: 'Legal NLP API',
    team: 'Client C - Legal Tech',
    model: 'HuggingFace BERT NER',
    endpoint: 'legal-nlp-ner-endpoint',
    namespace: 'client-c',
    url: import.meta.env.VITE_CLIENT_C_URL || 'http://k8s-clientc-legalnlp-0e7aa73795-8ad5f687f224569f.elb.us-east-1.amazonaws.com',
    featureCount: null,
    testPayload: null,
  },
]

function StatusBadge({ status }) {
  const colors = {
    healthy: 'bg-green-100 text-green-800',
    ready: 'bg-blue-100 text-blue-800',
    unreachable: 'bg-red-100 text-red-800',
    checking: 'bg-yellow-100 text-yellow-800',
    demo: 'bg-purple-100 text-purple-800',
  }
  return (
    <span className={`px-2 py-1 rounded-full text-xs font-medium ${colors[status] || colors.checking}`}>
      {status}
    </span>
  )
}

function ServiceCard({ service, healthStatus, demoMode }) {
  const [testResult, setTestResult] = useState(null)
  const [testing, setTesting] = useState(false)
  const [testInput, setTestInput] = useState(
    service.name === 'Legal NLP API'
      ? 'Acme Corp shall pay $50,000 to John Smith by December 31, 2025.'
      : ''
  )

  const runTest = async () => {
    setTesting(true)
    setTestResult(null)

    if (demoMode) {
      await new Promise(r => setTimeout(r, 800))
      setTestResult({ success: true, data: DEMO_RESPONSES[service.name].predict })
      setTesting(false)
      return
    }

    try {
      let body
      if (service.name === 'Legal NLP API') {
        body = JSON.stringify({ text: testInput })
      } else {
        body = JSON.stringify({ features: service.testPayload })
      }

      const res = await fetch(`${service.url}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      })
      const data = await res.json()
      setTestResult({ success: true, data })
    } catch (err) {
      setTestResult({ success: false, error: err.message })
    }
    setTesting(false)
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">{service.name}</h3>
          <p className="text-sm text-gray-500">{service.team}</p>
        </div>
        <StatusBadge status={healthStatus} />
      </div>

      <div className="space-y-2 text-sm text-gray-600 mb-4">
        <div className="flex justify-between">
          <span className="font-medium">Model:</span>
          <span>{service.model}</span>
        </div>
        <div className="flex justify-between">
          <span className="font-medium">Endpoint:</span>
          <span className="font-mono text-xs">{service.endpoint}</span>
        </div>
        <div className="flex justify-between">
          <span className="font-medium">Namespace:</span>
          <span className="font-mono text-xs">{service.namespace}</span>
        </div>
        {!demoMode && (
          <div className="flex justify-between">
            <span className="font-medium">Service URL:</span>
            <span className="font-mono text-xs">{service.url}</span>
          </div>
        )}
      </div>

      <div className="border-t pt-4">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-700">Test Request</span>
          <button
            onClick={runTest}
            disabled={testing}
            className="px-3 py-1 bg-indigo-600 text-white text-sm rounded hover:bg-indigo-700 disabled:opacity-50"
          >
            {testing ? 'Sending...' : 'Send'}
          </button>
        </div>

        {service.name === 'Legal NLP API' && (
          <textarea
            value={testInput}
            onChange={(e) => setTestInput(e.target.value)}
            className="w-full text-xs font-mono border rounded p-2 mb-2"
            rows={2}
          />
        )}

        {testResult && (
          <pre className={`text-xs p-2 rounded overflow-auto max-h-40 ${
            testResult.success ? 'bg-green-50 text-green-900' : 'bg-red-50 text-red-900'
          }`}>
            {testResult.success
              ? JSON.stringify(testResult.data, null, 2)
              : `Error: ${testResult.error}`}
          </pre>
        )}
      </div>
    </div>
  )
}

export default function App() {
  const [healthStatuses, setHealthStatuses] = useState({})
  const [lastPolled, setLastPolled] = useState(null)
  const [demoMode, setDemoMode] = useState(false)

  const checkHealth = async () => {
    const statuses = {}
    let allUnreachable = true
    for (const service of SERVICES) {
      try {
        const res = await fetch(`${service.url}/health`, { signal: AbortSignal.timeout(3000) })
        if (res.ok) {
          const data = await res.json()
          statuses[service.name] = data.status === 'healthy' ? 'healthy' : 'unreachable'
          if (data.status === 'healthy') allUnreachable = false
        } else {
          statuses[service.name] = 'unreachable'
        }
      } catch {
        statuses[service.name] = 'unreachable'
      }
    }

    if (allUnreachable) {
      setDemoMode(true)
      for (const service of SERVICES) {
        statuses[service.name] = 'demo'
      }
    } else {
      setDemoMode(false)
    }

    setHealthStatuses(statuses)
    setLastPolled(new Date().toLocaleTimeString())
  }

  useEffect(() => {
    checkHealth()
    const interval = setInterval(checkHealth, 15000)
    return () => clearInterval(interval)
  }, [])

  const healthyCount = Object.values(healthStatuses).filter(s => s === 'healthy').length
  const totalCount = SERVICES.length

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">ML Platform Ops Dashboard</h1>
              <p className="text-sm text-gray-500">Multi-client SageMaker + Kubernetes platform</p>
            </div>
            <div className="text-right">
              {demoMode ? (
                <div className="text-sm text-purple-600 font-medium">Demo Mode</div>
              ) : (
                <div className="text-sm text-gray-500">
                  Services: <span className="font-semibold text-green-600">{healthyCount}</span>/{totalCount} healthy
                </div>
              )}
              {lastPolled && (
                <div className="text-xs text-gray-400">
                  {demoMode ? 'Backends offline \u2014 showing sample data' : `Last polled: ${lastPolled} (every 15s)`}
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {demoMode && (
        <div className="bg-purple-50 border-b border-purple-200">
          <div className="max-w-7xl mx-auto px-4 py-2 sm:px-6 lg:px-8">
            <p className="text-xs text-purple-700 text-center">
              Services are not running. Showing demo data with sample predictions. Hit &ldquo;Send&rdquo; to see example responses.
            </p>
          </div>
        </div>
      )}

      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8 flex-1">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {SERVICES.map(service => (
            <ServiceCard
              key={service.name}
              service={service}
              healthStatus={healthStatuses[service.name] || 'checking'}
              demoMode={demoMode}
            />
          ))}
        </div>
      </main>

      <footer className="border-t bg-white">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <p className="text-xs text-gray-400 text-center">
            Built with FastAPI, SageMaker, Kubernetes (EKS), Terraform, GitHub Actions &amp; React
            {' \u00b7 '}
            <a href="https://github.com/kd365/assessment-iv" className="text-indigo-500 hover:underline" target="_blank" rel="noopener noreferrer">
              Source on GitHub
            </a>
          </p>
        </div>
      </footer>
    </div>
  )
}
