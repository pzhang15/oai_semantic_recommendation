import type { SearchItem } from '../lib/types'
import ProductCard from './ProductCard'

export default function ProductGrid({ items, onSelect, loading }: { items: SearchItem[], onSelect?: (item: SearchItem)=>void, loading?: boolean }) {
  return (
    <div className="relative">
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 opacity-100">
        {items.map(it => <ProductCard key={it.id} item={it} onSelect={onSelect} />)}
      </div>
      {loading && (
        <div className="absolute inset-0 bg-white/70 dark:bg-neutral-900/60 flex items-center justify-center">
          <div className="h-10 w-10 border-4 border-neutral-300 dark:border-neutral-700 border-t-blue-500 rounded-full animate-spin" />
        </div>
      )}
    </div>
  )
}


