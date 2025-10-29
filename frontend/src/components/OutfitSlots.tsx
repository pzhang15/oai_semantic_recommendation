import type { OutfitResponse } from '../lib/types'
import { fmtPrice } from '../lib/format'

export default function OutfitSlots({ data }: { data: OutfitResponse }) {
  const entries = Object.entries(data.slots)
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {entries.map(([slot, items]) => (
        <div key={slot} className="border rounded p-3">
          <div className="font-semibold text-sm mb-2">{slot}</div>
          <div className="space-y-2">
            {items.map((it, idx) => (
              <a key={it.id+idx} className="block border rounded p-2 hover:bg-neutral-50 dark:hover:bg-neutral-900" href={it.product_url || '#'} target="_blank" rel="noreferrer">
                <div className="text-sm line-clamp-2">{it.title}</div>
                <div className="text-xs text-neutral-500">{it.brand}</div>
                <div className="text-xs flex items-center justify-between">
                  <span>{fmtPrice(it.price)}</span>
                  {it.why && <span title={it.why} className="text-[11px] text-neutral-500 line-clamp-1">{it.why}</span>}
                </div>
              </a>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}


