# before running this script, set the config files as follows:
# configs/tsc/base.yml -> episodes: 100
# configs/tsc/presslight.yml -> train_model: True

agent=presslight
worlds=(cityflow sumo)
networks=( tempe_1x1 bullhead_1 cologne1 ingolstadt1 hz1x1 tempe_16 bullhead_3 cologne3 ingolstadt7 hz4x4 )

for network in "${networks[@]}"; do
  for world in "${worlds[@]}"; do
    echo "Running: python run.py --agent $agent --world $world --network $network"
    python run.py --agent "$agent" --world "$world" --network "$network"
  done
done

