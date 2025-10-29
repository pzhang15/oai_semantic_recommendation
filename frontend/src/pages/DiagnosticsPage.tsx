import { useQuery } from '@tanstack/react-query'
import { getHealth, getStats } from '../lib/api'

export default function DiagnosticsPage() {
  const h = useQuery({ queryKey: ['health'], queryFn: getHealth })
  const s = useQuery({ queryKey: ['stats'], queryFn: getStats, refetchInterval: 5000 })
  return (
    <div className="max-w-6xl mx-auto px-4 py-6 space-y-6">
      <section>
        <h2 className="font-semibold mb-2">Health</h2>
        {h.data ? (
          <div className="text-sm">{h.data.backend} dim={h.data.dim} size={h.data.size} ({h.data.index_type})</div>
        ) : <div>Loading...</div>}
      </section>
      <section>
        <h2 className="font-semibold mb-2">Stats</h2>
        {s.data ? (
          <pre className="text-xs bg-neutral-100 dark:bg-neutral-900 p-2 rounded overflow-auto max-h-96">{JSON.stringify(s.data, null, 2)}</pre>
        ) : <div>Loading...</div>}
      </section>
    </div>
  )
}


