# Run via: bundle exec rails runner "load 'script/enable_premium_features.rb'"
#
# One-time script to enable all premium features on existing accounts.
# Safe to run multiple times (idempotent).

premium_features = %w[
  custom_roles sla audit_logs captain_integration captain_tasks
  disable_branding help_center custom_tools captain_integration_v2
  advanced_assignment saml conversation_required_attributes
  csat_review_notes
]

Account.find_each do |account|
  account.enable_features!(*premium_features)
  puts "✓ Enabled premium features for Account ##{account.id} (#{account.name})"
end

# Set the installation pricing plan to 'enterprise' in the database
config = InstallationConfig.find_or_create_by!(name: 'INSTALLATION_PRICING_PLAN') do |c|
  c.value = 'enterprise'
end
config.update!(value: 'enterprise')
puts "\n✓ INSTALLATION_PRICING_PLAN set to 'enterprise'"

puts "\nDone! All accounts now have full enterprise features."
