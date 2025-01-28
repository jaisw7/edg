echo "np.array(["

prefix="periodic/periodic"
geo_file="examples/advection/${prefix}.geo"
msh_file="examples/advection/${prefix}.msh"
ini_file="examples/advection/${prefix}.ini"

for degree in {4..4}; do
    gmsh -2 "${geo_file}" >/dev/null
    err=()
    for i in {1..5}; do
        edgfsAdv2D run "${ini_file}" "${msh_file}" -v basis-tri::degree $degree -v time::dt $(echo "0.01/($i)" | bc -l) >log 2>&1
        err+="$(cat log | tail -n 1), "
        gmsh -refine "${msh_file}" >/dev/null
    done
    echo "[$err], "
done
echo "], dtype=np.float64)"
