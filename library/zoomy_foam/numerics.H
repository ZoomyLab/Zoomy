#include "List.H"
#include "Map.H"
#include "UList.H"
#include "VectorSpace.H"
#include "fvMesh.H"
#include "label.H"
#include "messageStream.H"
#include "polyPatch.H"
#include "scalar.H"
#include "vector.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "scalarField.H"
#include "Model.H"
#include "word.H"

namespace numerics
{

Foam::HashTable<Foam::label, Foam::word> get_map_patchNameToPatchIndex()
{
    Foam::HashTable<Foam::label, Foam::word> patchNameToPatchIndex;
    forAll(Model::map_boundary_tag_to_function_index, patchI)
    {
        patchNameToPatchIndex.insert(Model::map_boundary_tag_to_function_index[patchI], patchI);
    }
    return patchNameToPatchIndex;
}


Foam::tmp<Foam::volScalarField> quasilinear_operator
(
    const Foam::surfaceScalarField& Dp,
    const Foam::surfaceScalarField& Dm
)
{
    const Foam::fvMesh& mesh = Dp.mesh();

    Foam::tmp<Foam::volScalarField> tResult
    (
        new Foam::volScalarField
        (
            Foam::IOobject
            (
                "quasilinear_operator",
                Dp.mesh().time().name(),
                Dp.mesh(),
                Foam::IOobject::NO_READ,
                Foam::IOobject::NO_WRITE
            ),
            Dp.mesh(),
            Foam::dimensionedScalar("zero", Dp.dimensions()/Foam::dimVolume, 0.0)
        )
    );

    Foam::scalarField& resultI = tResult.ref().primitiveFieldRef();

    // iterate over internal faces
    for (int faceI = 0; faceI < mesh.nInternalFaces(); ++faceI) {
        const Foam::label& own = mesh.faceOwner()[faceI];
        const Foam::label& nei = mesh.faceNeighbour()[faceI];

        resultI[own] += (Dm)[faceI];
        resultI[nei] += (Dp)[faceI]; 
    }

    forAll(mesh.boundaryMesh(), patchI)
    {
        auto patchNameToPatchIndex = get_map_patchNameToPatchIndex();
        const Foam::polyPatch& p = mesh.boundaryMesh()[patchI];
    
        if (patchNameToPatchIndex.found(p.name()))
        {
            forAll(p, localFaceI)
            {
                const Foam::label cellI = p.faceCells()[localFaceI];
                resultI[cellI] += Dm.boundaryField()[patchI][localFaceI];
            }
        }
    }

    resultI /= mesh.V();
    return tResult;
}



// set Q to enforce zero gradient
inline void correctBoundaryQ
(
    const Foam::List<Foam::volScalarField*>& Q,
    const Foam::List<Foam::volScalarField*>& Qaux,
    const Foam::scalar& time
)
{

    auto patchNameToPatchIndex = get_map_patchNameToPatchIndex();
    const Foam::fvMesh& mesh = Q[0]->mesh();
    forAll(mesh.boundaryMesh(), foamPatchI)
    {
        const Foam::polyPatch& p = mesh.boundaryMesh()[foamPatchI];
        if (patchNameToPatchIndex.found(p.name()))
        {
            const Foam::label patchI = patchNameToPatchIndex[p.name()];

            forAll(p, localFaceI)
            {
                const Foam::label& cellI = p.faceCells()[localFaceI];
                const Foam::vector& n = p.faceNormals()[localFaceI];
                const Foam::vector& X = mesh.cellCentres()[cellI];
                const Foam::scalar dX = Foam::mag(p.faceCentres()[localFaceI] - mesh.cellCentres()[cellI]);

                Foam::List<Foam::scalar> localQ(Model::n_dof_q);
                forAll(Q, QI)
                {
                    localQ[QI]= (*Q[QI])[cellI];
                }

                Foam::List<Foam::scalar> localQaux(Model::n_dof_qaux);
                forAll(Qaux, QauxI)
                {
                    localQaux[QauxI]= (*Qaux[QauxI])[cellI];
                }

                const Foam::List<Foam::List<Foam::scalar>> bc =
                    Model::boundary_conditions(localQ, localQaux, n, X, time, dX);

                forAll(Q, QI)
                {
                    Q[QI]->boundaryFieldRef()[foamPatchI][localFaceI] = bc[patchI][QI];
                }
            }
        }
    }
}

inline Foam::scalar max_abs_eigenvalue
(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::vector& n)
{
    Foam::scalar max_abs_ev = -Foam::GREAT;
    auto evs = Model::eigenvalues(Q, Qaux, n);
    forAll(evs, e)
    {
        if (std::abs(e) > max_abs_ev) max_abs_ev = std::abs(e);
    }
    return max_abs_ev;
}

inline void localNumericalFlux
(
    const Foam::List<Foam::scalar>& Ql,
    const Foam::List<Foam::scalar>& Qr,
    const Foam::List<Foam::scalar>& Qauxl,
    const Foam::List<Foam::scalar>& Qauxr,
    const Foam::vector& n,
    Foam::List<Foam::scalar>& Dp,
    Foam::List<Foam::scalar>& Dm
)
{
    const Foam::List<Foam::List<Foam::scalar>> flux_x_l = Model::flux_x(Ql, Qauxl);
    const Foam::List<Foam::List<Foam::scalar>> flux_y_l = Model::flux_y(Ql, Qauxl);
    const Foam::List<Foam::List<Foam::scalar>> flux_x_r = Model::flux_x(Qr, Qauxr);
    const Foam::List<Foam::List<Foam::scalar>> flux_y_r = Model::flux_y(Qr, Qauxr);

    Foam::scalar lam_l = max_abs_eigenvalue(Ql, Qauxl, n);
    Foam::scalar lam_r = max_abs_eigenvalue(Qr, Qauxr, n);
    Foam::scalar lam = std::max(lam_l, lam_r);

    Foam::List<Foam::scalar> flux (Model::n_dof_q);

    for (int i = 0; i < Model::n_dof_q; ++i) 
    {
        flux[i] = 0.5 * (flux_x_l[i][0] * n.x() + flux_y_l[i][0] * n.y() + flux_x_r[i][0] * n.x() + flux_y_r[i][0] * n.y()) - 0.5 * lam * (Qr[i] - Ql[i]);
    }
    for (int i = 0; i < Model::n_dof_q; ++i) 
    {
        Dm[i] = flux[i];
        Dp[i] = -flux[i];
    }
    return;
}

inline void localNumericalQuasilinearFlux
(
    const Foam::List<Foam::scalar>& Ql,
    const Foam::List<Foam::scalar>& Qr,
    const Foam::List<Foam::scalar>& Qauxl,
    const Foam::List<Foam::scalar>& Qauxr,
    const Foam::vector& n,
    Foam::List<Foam::scalar>& Dp,
    Foam::List<Foam::scalar>& Dm
)
{
        
    Foam::FixedList<Foam::scalar, 3> wi
    (
        {
            5.0/18.0,  
            4.0/9.0,   
            5.0/18.0   
        }
    );
    Foam::FixedList<Foam::scalar, 3> xi
    (
        {
            0.5*(1.0 - sqrt(3.0/5.0)),  
            0.5,                        
            0.5*(1.0 + sqrt(3.0/5.0))   
        }
    );

    Foam::List<Foam::List<Foam::scalar>> A
    (
        Model::n_dof_q,
        Foam::List<Foam::scalar>(Model::n_dof_q, 0.0)
    );
    Foam::List<Foam::List<Foam::scalar>> I
    (
        Model::n_dof_q,
        Foam::List<Foam::scalar>(Model::n_dof_q, 0.0)
    );

    for (int i=0; i < Model::n_dof_q; ++i) I[i][i] = 1.0;

    Foam::List<Foam::scalar> Qi(Model::n_dof_q, 0.0);
    Foam::List<Foam::scalar> Qauxi(Model::n_dof_qaux, 0.0);
    for (int i=0; i < 3; ++i)
    {
        for (int j=0; j<Model::n_dof_q; ++j) Qi[j] = Ql[j] + (Qr[j]-Ql[j]) * xi[i];
        for (int j=0; j<Model::n_dof_qaux; ++j) Qauxi[j] = Qauxl[j] + (Qauxr[j]-Qauxl[j]) * xi[i];
        //const Foam::List<Foam::List<Foam::scalar>> Ax = Model::nonconservative_matrix_x(Qi, Qauxi);
        //const Foam::List<Foam::List<Foam::scalar>> Ay = Model::nonconservative_matrix_y(Qi, Qauxi);
        const Foam::List<Foam::List<Foam::scalar>> Ax = Model::quasilinear_matrix_x(Qi, Qauxi);
        const Foam::List<Foam::List<Foam::scalar>> Ay = Model::quasilinear_matrix_y(Qi, Qauxi);
        for (int l=0; l < Model::n_dof_q; ++l)
        {
            for (int k=0; k < Model::n_dof_q; ++k)
            {
                A[l][k] += wi[i] * (Ax[l][k] * n.x() + Ay[l][k] * n.y());
            }
        }
    }

    Foam::scalar lam_l = max_abs_eigenvalue(Ql, Qauxl, n);
    Foam::scalar lam_r = max_abs_eigenvalue(Qr, Qauxr, n);
    Foam::scalar lam = std::max(lam_l, lam_r);

    for (int i = 0; i < Model::n_dof_q; ++i) 
    {
        Dp[i] = 0;
        Dm[i] = 0;
    }

    for (int i = 0; i < Model::n_dof_q; ++i) 
    {
        for (int j = 0; j < Model::n_dof_q; ++j) 
        {
            Dp[i] += 0.5 * (A[i][j]  + lam * I[i][j])* (Qr[j] - Ql[j]);
            Dm[i] += 0.5 * (A[i][j]  - lam * I[i][j])* (Qr[j] - Ql[j]);
        }
    }
    return;
}

inline void updateNumericalQuasilinearFlux
(
    Foam::List<Foam::surfaceScalarField*>& Dp,
    Foam::List<Foam::surfaceScalarField*>& Dm,
    const Foam::List<Foam::volScalarField*>& Q,
    const Foam::List<Foam::volScalarField*>& Qaux
)
{
    const Foam::fvMesh& mesh = Q[0]->mesh();

    Foam::List<Foam::scalar> dp(Model::n_dof_q, 0.), dm(Model::n_dof_q, 0.);
    Foam::List<Foam::scalar> fp(Model::n_dof_q, 0.), fm(Model::n_dof_q, 0.);


    // iterate over internal faces
    for (int faceI = 0; faceI < mesh.nInternalFaces(); ++faceI) {
        const Foam::vector n = mesh.Sf()[faceI] / mesh.magSf()[faceI];
        const Foam::label& own = mesh.faceOwner()[faceI];
        const Foam::label& nei = mesh.faceNeighbour()[faceI];

        Foam::List<Foam::scalar> Ql(Model::n_dof_q), Qr(Model::n_dof_q);
        Foam::List<Foam::scalar> Qauxl(Model::n_dof_qaux), Qauxr(Model::n_dof_qaux);
        forAll(Q, QI)
        {
            Ql[QI] = (*Q[QI])[own];
            Qr[QI] = (*Q[QI])[nei];
        }
        forAll(Qaux, QauxI)
        {
            Qauxl[QauxI] = (*Qaux[QauxI])[own];
            Qauxr[QauxI] = (*Qaux[QauxI])[nei];
        }

        localNumericalQuasilinearFlux(Ql, Qr, Qauxl, Qauxr, n, dp, dm);
        //localNumericalFlux(Ql, Qr, Qauxl, Qauxr, n, fp, fm);
        forAll(Dp, i)
        {
            (*Dp[i])[faceI] = (dp[i] + fp[i]) * mesh.magSf()[faceI];
            (*Dm[i])[faceI] = (dm[i] + fm[i]) * mesh.magSf()[faceI];
        }
    }

    // iterate over external faces
    //
    const Foam::polyMesh& pMesh = mesh;

    forAll(mesh.boundaryMesh(), patchI)
    {
        auto patchNameToPatchIndex = get_map_patchNameToPatchIndex();
        const Foam::polyPatch& p = mesh.boundaryMesh()[patchI];

        if (patchNameToPatchIndex.found(p.name()))
        {
            forAll(p, localFaceI)
            {
                const Foam::label& cellI = p.faceCells()[localFaceI];
                const Foam::vector& n = p.faceNormals()[localFaceI];
                Foam::List<Foam::scalar> Ql(Model::n_dof_q);
                Foam::List<Foam::scalar> Qauxl(Model::n_dof_qaux);
                Foam::List<Foam::scalar> Qr(Model::n_dof_q);
                Foam::List<Foam::scalar> Qauxr(Model::n_dof_qaux);
                forAll(Q, QI)
                {
                    Ql[QI] = (*Q[QI])[cellI];
                    Qr[QI] = Q[QI]->boundaryFieldRef()[patchI][localFaceI];
                }
                forAll(Qaux, QauxI)
                {
                    Qauxl[QauxI] = (*Qaux[QauxI])[cellI];
                    Qauxr[QauxI] = Qaux[QauxI]->boundaryFieldRef()[patchI][localFaceI];
                }


                localNumericalQuasilinearFlux(Ql, Qr, Qauxl, Qauxr, n, dp, dm);
                //localNumericalFlux(Ql, Qr, Qauxl, Qauxr, n, fp, fm);
                forAll(Dm, i)
                {
                    Dm[i]->boundaryFieldRef()[patchI][localFaceI] = (dm[i] + fm[i]) * pMesh.magFaceAreas()[p.start() + localFaceI];
                    Dp[i]->boundaryFieldRef()[patchI][localFaceI] = (dp[i] + fp[i]) * pMesh.magFaceAreas()[p.start() + localFaceI];
                }
            }
        }
    }
}

Foam::surfaceScalarField computeFaceMinInradius(const Foam::fvMesh& mesh, const Foam::Time& runTime)
{
    Foam::surfaceScalarField faceMinInradius
    (
        Foam::IOobject
        (
            "faceMinInradius",
            runTime.name(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh,
        Foam::dimensionedScalar("zero", Foam::dimLength, 0.0)
    );

    const Foam::polyMesh& pMesh = mesh;
    const Foam::pointField& points = pMesh.points();
    const Foam::cellList& cells = mesh.cells();

    Foam::scalarField cellInradius(mesh.nCells(), 0.0);

    forAll(cells, celli)
    {
        const Foam::cell& c = cells[celli];
        const Foam::faceList& faces = pMesh.faces();

        // Identify the "bottom" face: assume it's the one with smallest average z
        Foam::label bottomFaceI = -1;
        Foam::scalar minAvgZ = Foam::GREAT;

        forAll(c, fi)
        {
            const Foam::face& f = faces[c[fi]];
            Foam::scalar avgZ = 0.0;
            forAll(f, vi)
            {
                avgZ += points[f[vi]].z();
            }
            avgZ /= f.size();

            if (avgZ < minAvgZ)
            {
                minAvgZ = avgZ;
                bottomFaceI = c[fi];
            }
        }

        // Now get the 4 bottom vertices (assuming rectangular)
        const Foam::face& bottomFace = faces[bottomFaceI];
        const Foam::labelList& vertLabels = bottomFace;

        if (vertLabels.size() != 4)
        {
            FatalErrorInFunction
                << "Cell " << celli
                << " has a bottom face with " << vertLabels.size()
                << " vertices instead of 4. "
                << "This function assumes rectangular base faces."
                << Foam::nl << exit(Foam::FatalError);
        }

        // Coordinates of bottom vertices
        const Foam::point& p0 = points[vertLabels[0]];
        const Foam::point& p1 = points[vertLabels[1]];
        const Foam::point& p2 = points[vertLabels[2]];
        const Foam::point& p3 = points[vertLabels[3]];

        // Compute side lengths (assuming rectangle ordering)
        Foam::scalar a = Foam::mag(p1 - p0);
        Foam::scalar b = Foam::mag(p3 - p0);

        Foam::scalar rin = 0.5 * Foam::min(a, b);
        cellInradius[celli] = rin;
    }

    const Foam::labelUList& own = mesh.owner();
    const Foam::labelUList& nei = mesh.neighbour();

    Foam::scalarField& faceValues = faceMinInradius.primitiveFieldRef();

    forAll(faceValues, faceI)
    {
        int ownCell = own[faceI];
        int neiCell = (faceI < nei.size()) ? nei[faceI] : ownCell;

        Foam::scalar rOwn = cellInradius[ownCell];
        Foam::scalar rNei = cellInradius[neiCell];

        faceValues[faceI] = Foam::min(rOwn, rNei);
    }

    return faceMinInradius;
}

Foam::scalar compute_dt(
    const Foam::List<Foam::volScalarField*>& Q, 
    const Foam::List<Foam::volScalarField*>& Qaux, 
    const Foam::surfaceScalarField& min_inradius, 
    Foam::scalar Co)
{
    const Foam::fvMesh &mesh = Q[0]->mesh();
    const Foam::surfaceVectorField& Sf = mesh.Sf();
    Foam::surfaceVectorField n (mesh.Sf()/mesh.magSf());

    Foam::scalar dt = Foam::GREAT;

    forAll(Sf, faceI)
    {
        Foam::scalar max_abs_ev = 0.0;
        Foam::List<Foam::scalar> q(Model::n_dof_q);
        Foam::List<Foam::scalar> qaux(Model::n_dof_qaux);
        for(int j = 0; j < Model::n_dof_q; ++j)
        {
            q[j]= (*Q[j])[mesh.faceOwner()[faceI]];
        }
        for(int j = 0; j < Model::n_dof_qaux; ++j)
        {
            qaux[j]= (*Qaux[j])[mesh.faceOwner()[faceI]];
        }
        auto evs = Model::eigenvalues(q, qaux, n[faceI]);
        for(int j = 0; j < Model::n_dof_q; ++j)
        {
            max_abs_ev = std::max(max_abs_ev, std::abs(evs[j][0]));
        }
        const Foam::scalar dtCell = Co * min_inradius[faceI] / max_abs_ev;
        dt = std::min(dt, dtCell);
    }
    dt = std::max(dt, 0.0000001);

    return dt;
}

} //end namespace numerics
