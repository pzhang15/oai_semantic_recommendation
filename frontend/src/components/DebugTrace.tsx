export default function DebugTrace({ trace, e2eMs }: { trace?: any; e2eMs?: number }) {
  if (!trace) return null
  const timings = trace?.timings || {}
  const keys = ["parse_ms", "retrieve_ms", "filter_ms", "judge_ms", "mmr_ms"]
  const hasTimings = keys.some(k => typeof timings[k] === 'number')
  const total = keys.reduce((s, k) => s + (typeof timings[k] === 'number' ? timings[k] : 0), 0)
  const rtim = trace?.retrieval?.timings || {}
  const htim = trace?.retrieval?.hybrid?.timings || {}
  const judge = trace?.judge || {}
  const gate = trace?.judge_gate || judge?.gate
  const lexScores = trace?.retrieval?.hybrid?.lex_scores
  const gateCost = gate?.cost
  const expansion = trace?.retrieval?.hybrid?.expansion
  return (
    <details className="mt-4" open>
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
          {(htim.lex_load_ms || htim.lex_compose_ms || htim.lex_transform_ms || htim.lex_matvec_ms || htim.lex_topk_ms || htim.fuse_ms) && (
            <div className="mt-3">
              <div className="text-neutral-600 dark:text-neutral-300 mb-1">Hybrid (TF-IDF / RRF) breakdown</div>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2">
                {['lex_load_ms','lex_compose_ms','lex_transform_ms','lex_matvec_ms','lex_topk_ms','fuse_ms'].map(k => (
                  <div key={k} className="border rounded p-2">
                    <div className="text-neutral-500">{k.replace('_ms','')}</div>
                    <div>{typeof htim[k] === 'number' ? `${htim[k].toFixed(2)} ms` : '—'}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
          {gate && (
            <div className="mt-3">
              <div className="text-neutral-600 dark:text-neutral-300 mb-1">Judge gating</div>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2 text-xs">
                {[
                  {k:'mode', v: gate.mode},
                  {k:'top_m', v: gate.top_m},
                  {k:'reason', v: gate.reason},
                  {k:'cache_hit', v: gate.cache?.hit},
                  {k:'cache_age_s', v: gate.cache?.age_secs},
                  {k:'escalated', v: gate.escalated},
                  {k:'llm_called', v: judge?.llm_called},
                  {k:'llm_skip_reason', v: judge?.llm_skip_reason},
                ].map(({k,v}) => (
                  <div key={k} className="border rounded p-2">
                    <div className="text-neutral-500">{k}</div>
                    <div>{v === undefined || v === null ? '—' : String(v)}</div>
                  </div>
                ))}
              </div>
              {gateCost && (
                <div className="mt-2 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2 text-xs">
                  {[
                    {k:'projected_req_usd', v: gateCost.projected_req_usd},
                    {k:'per_req_cap', v: gateCost.per_req_cap},
                    {k:'spent_today_usd', v: gateCost.spent_today_usd},
                    {k:'daily_cap', v: gateCost.daily_cap},
                  ].map(({k,v}) => (
                    <div key={k} className="border rounded p-2">
                      <div className="text-neutral-500">{k}</div>
                      <div>{v === undefined || v === null ? '—' : String(v)}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          {lexScores && Object.keys(lexScores).length > 0 && (
            <div className="mt-3">
              <div className="text-neutral-600 dark:text-neutral-300 mb-1">Lexical scores (top 10)</div>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2 text-xs">
                {Object.entries(lexScores)
                  .sort((a:any,b:any) => (b[1] ?? 0) - (a[1] ?? 0))
                  .slice(0, 10)
                  .map(([pid, sc]: any) => (
                    <div key={pid as string} className="border rounded p-2">
                      <div className="text-neutral-500">{String(pid)}</div>
                      <div>{typeof sc === 'number' ? sc.toFixed(4) : String(sc)}</div>
                    </div>
                ))}
              </div>
            </div>
          )}
          {expansion && (
            <div className="mt-3">
              <div className="text-neutral-600 dark:text-neutral-300 mb-1">Micro query expansion</div>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2 text-xs">
                {[{k:'enabled', v: expansion.enabled}, {k:'reason', v: expansion.reason}, {k:'time_budget_ms', v: expansion.time_budget_ms}, {k:'faiss_ms', v: expansion.faiss_ms}, {k:'num_det', v: expansion.num_det}]
                  .filter(x => x.v !== undefined)
                  .map(({k,v}) => (
                    <div key={k} className="border rounded p-2">
                      <div className="text-neutral-500">{k}</div>
                      <div>{String(v)}</div>
                    </div>
                ))}
              </div>
              <div className="mt-2 text-xs">
                <div className="text-neutral-500">base</div>
                <div className="border rounded p-2 break-words">{expansion.base_query || '—'}</div>
              </div>
              {Array.isArray(expansion.det_queries) && expansion.det_queries.length > 0 && (
                <div className="mt-2 text-xs">
                  <div className="text-neutral-500">deterministic rewrites</div>
                  <ul className="list-disc ml-6">
                    {expansion.det_queries.map((q: string, i: number) => (
                      <li key={i} className="break-words">{q}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
      <pre className="text-xs bg-neutral-100 dark:bg-neutral-900 p-2 rounded mt-2 overflow-auto max-h-64">{JSON.stringify(trace, null, 2)}</pre>
    </details>
  )
}


