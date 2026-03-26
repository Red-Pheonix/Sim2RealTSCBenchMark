agent=presslight
gat_model=(gat ugat)
networks=( tempe_1x1 bullhead_1 cologne1 ingolstadt1 hz1x1 )
real_settings=( setting1 setting2 setting3 setting4 )

for network in "${networks[@]}"; do
  for real_setting in "${real_settings[@]}"; do
    echo "Running: python run_s2r.py --agent $agent --network $network --gat_model $gat_model --real_setting $real_setting"
    python run_s2r.py --agent "$agent" --network "$network" --gat_model "$gat_model" --real_setting "$real_setting"
  done
done
