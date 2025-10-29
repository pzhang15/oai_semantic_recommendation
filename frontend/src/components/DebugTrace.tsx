export default function DebugTrace({ trace, e2eMs }: { trace?: any; e2eMs?: number }) {
  if (!trace) return null
  const timings = trace?.timings || {}
  const keys = ["parse_ms", "retrieve_ms", "filter_ms", "judge_ms", "mmr_ms"]
  const hasTimings = keys.some(k => typeof timings[k] === 'number')
  const total = keys.reduce((s, k) => s + (typeof timings[k] === 'number' ? timings[k] : 0), 0)
  const rtim = trace?.retrieval?.timings || {}
  const htim = trace?.retrieval?.hybrid?.timings || {}
  return (
    <details className="mt-4">
      <summary className="cursor-pointer text-sm">Debug trace</summary>
      {hasTimings && (
        <div className="mt-2 text-xs">
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2">
            {keys.map(k => (
              <div key={k} className="border rounded p-2">
                <div className="text-neutral-500">{k.replace('_ms','')}</div>
                <div>{typeof timings[k] === 'number' ? `${timings[k].toFixed(2)} ms` : '—'}</div>
              </div>
            ))}
            <div className="border rounded p-2">
              <div className="text-neutral-500">total_components</div>
              <div>{total.toFixed(2)} ms</div>
            </div>
            <div className="border rounded p-2">
              <div className="text-neutral-500">e2e_client</div>
              <div>{typeof e2eMs === 'number' ? `${e2eMs.toFixed(2)} ms` : '—'}</div>
            </div>
          </div>
          {(rtim.embed_ms || rtim.faiss_ms || rtim.join_ms) && (
            <div className="mt-3">
              <div className="text-neutral-600 dark:text-neutral-300 mb-1">Retrieval breakdown</div>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2">
                {['embed_ms','faiss_ms','join_ms'].map(k => (
                  <div key={k} className="border rounded p-2">
                    <div className="text-neutral-500">{k.replace('_ms','')}</div>
                    <div>{typeof rtim[k] === 'number' ? `${rtim[k].toFixed(2)} ms` : '—'}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
          {(htim.lex_compose_ms || htim.lex_transform_ms || htim.lex_matvec_ms || htim.lex_topk_ms || htim.fuse_ms) && (
            <div className="mt-3">
              <div className="text-neutral-600 dark:text-neutral-300 mb-1">Hybrid (TF-IDF / RRF) breakdown</div>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2">
                {['lex_compose_ms','lex_transform_ms','lex_matvec_ms','lex_topk_ms','fuse_ms'].map(k => (
                  <div key={k} className="border rounded p-2">
                    <div className="text-neutral-500">{k.replace('_ms','')}</div>
                    <div>{typeof htim[k] === 'number' ? `${htim[k].toFixed(2)} ms` : '—'}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
      <pre className="text-xs bg-neutral-100 dark:bg-neutral-900 p-2 rounded mt-2 overflow-auto max-h-64">{JSON.stringify(trace, null, 2)}</pre>
    </details>
  )
}


