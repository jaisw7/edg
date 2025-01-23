echo "np.array(["
for degree in {1..4}; do
    gmsh -2 examples/advection/periodic/periodic.geo >/dev/null
    err=()
    for i in {1..6}; do
        edgfsAdv2D run ./examples/advection/periodic/periodic.ini ./examples/advection/periodic/periodic.msh -v basis-tri::degree $degree -v time::dt $(echo "0.01/($i*$i)" | bc -l) >log 2>&1
        err+="$(cat log | tail -n 1), "
        gmsh -refine examples/advection/periodic/periodic.msh >/dev/null
    done
    echo "[$err], "
done
echo "], dtype=np.float64)"
