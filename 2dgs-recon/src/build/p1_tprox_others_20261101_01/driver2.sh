cd "<PROJECT_ROOT>"
L=build/p1_tprox_others_20261101_01/driver.log; echo "=== driver2 (infra_retry) $(date +%T)" >> $L
repush() { echo "== infra_retry push $1 $(date +%T)" >> $L; python -X utf8 -m tomato3d.organs.skeleton_tprox --step push --id $1 >> $L 2>&1 || echo "FAIL push $1" >> $L; }
push() { echo "== push $1 $(date +%T)" >> $L; (python -X utf8 -m tomato3d.organs.skeleton_tprox --step prep --id $1 && python -X utf8 -m tomato3d.organs.skeleton_tprox --step push --id $1) >> $L 2>&1 || echo "FAIL push $1" >> $L; }
for p in t2_02 e_01 e_04 e_05; do repush $p; done
set -- n_01 n_02 n_03 n_05 n_06 t2_05
for p in t2_02 e_01 e_04 e_05 n_01 n_02 n_03 n_05 n_06 t2_05; do
  echo "== finish $p $(date +%T)" >> $L; python -X utf8 -m tomato3d.organs.skeleton_tprox --step finish --id $p >> $L 2>&1 && echo "DONE $p" >> $L || echo "FAIL finish $p" >> $L
  if [ -n "$1" ]; then push $1; shift; fi
done
echo ALLDONE >> $L
