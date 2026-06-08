task :before_assets_precompile do
  system('pnpm install')
  system('echo "-------------- Bulding SDK for Production --------------"')
  system('pnpm run build:sdk')
  system('echo "-------------- Bulding App for Production --------------"')
end

Rake::Task['assets:precompile'].enhance %w[before_assets_precompile]
