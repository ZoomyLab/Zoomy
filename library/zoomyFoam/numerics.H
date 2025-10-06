#include "List.H"
#include "fvMesh.H"
#include "polyPatch.H"
#include "scalar.H"
#include "vector.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "scalarField.H"
#include "Model.h"

namespace numerics
{

// set Q to enforce zero gradient
inline void correctBoundaryQ
(
    Foam::List<Foam::volScalarField*>& Q
)
{
    const Foam::fvMesh& mesh = Q[0]->mesh();
    forAll(mesh.boundaryMesh(), patchI)
    {
        const Foam::polyPatch& p = mesh.boundaryMesh()[patchI];
        forAll(p, localFaceI)
        {
            const Foam::label& cellI = p.faceCells()[localFaceI];
            forAll(Q, QI)
            {
                Q[QI]->boundaryFieldRef()[patchI][localFaceI] = (*Q[QI])[cellI];
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

inline Foam::List<Foam::scalar> localNumericalQuasilinearFlux
(
    const Foam::List<Foam::scalar>& Ql,
    const Foam::List<Foam::scalar>& Qr,
    const Foam::List<Foam::scalar>& Qauxl,
    const Foam::List<Foam::scalar>& Qauxr,
    const Foam::vector& n
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
        const Foam::List<Foam::List<Foam::scalar>> Ax = Model::quasilinear_matrix_x(Qi, Qauxi);
        const Foam::List<Foam::List<Foam::scalar>> Ay = Model::quasilinear_matrix_y(Qi, Qauxi);
        for (int k=0; k < Model::n_dof_q; ++k)
        {
            for (int l=0; l < Model::n_dof_q; ++l)
            {
                A[l][k] += wi[i] * (Ax[l][k] * n.x() + Ay[l][k] * n.y());
            }
        }
    }

    Foam::scalar lam_l = max_abs_eigenvalue(Ql, Qauxl, n);
    Foam::scalar lam_r = max_abs_eigenvalue(Qr, Qauxr, n);
    Foam::scalar lam = std::max(lam_l, lam_r);

    Foam::List<Foam::scalar> flux (Model::n_dof_q);

    for (int i = 0; i < Model::n_dof_q; ++i) 
    {
        for (int j = 0; i < Model::n_dof_q; ++j) 
        {
            flux[i] = 0.5 * (A[i][j]  - lam * I[i][j])* (Qr[i] - Ql[i]);
        }
    }
    return flux;
}

inline Foam::List<Foam::scalar> localNumericalFlux
(
    const Foam::List<Foam::scalar>& Ql,
    const Foam::List<Foam::scalar>& Qr,
    const Foam::List<Foam::scalar>& Qauxl,
    const Foam::List<Foam::scalar>& Qauxr,
    const Foam::vector& n
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
        flux[i] = 0.5 * (flux_x_l[i][0] * n.x() + flux_y_l[i][0] * n.y() + flux_x_r[i][0] * n.x() + flux_y_r[i][0] * n.y()) - lam * (Qr[i] - Ql[i]);
    }
    return flux;
}

inline void updateNumericalQuasilinearFlux
(
    Foam::List<Foam::surfaceScalarField*>& F_in,
    Foam::List<Foam::surfaceScalarField*>& F_out,
    const Foam::List<Foam::volScalarField*>& Q,
    const Foam::List<Foam::volScalarField*>& Qaux
)
{
    const Foam::fvMesh& mesh = Q[0]->mesh();

    // iterate over internal faces
    for (int faceI = 0; faceI < mesh.nInternalFaces(); ++faceI) {
        const Foam::vector n = mesh.Sf()[faceI] / mesh.magSf()[faceI];
        const Foam::label& own = mesh.faceOwner()[faceI];
        const Foam::label& nei = mesh.faceNeighbour()[faceI];

        Foam::List<Foam::scalar> Ql(Q.size()), Qr(Q.size());
        Foam::List<Foam::scalar> Qauxl(Qaux.size()), Qauxr(Qaux.size());
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

        const Foam::List<Foam::scalar> flux_in = localNumericalFlux(Ql, Qr, Qauxl, Qauxr, n);
        const Foam::List<Foam::scalar> flux_out = localNumericalFlux(Qr, Ql, Qauxr, Qauxl, n);
        forAll(F_in, FI)
        {
            (*F_in[FI])[faceI] = flux_in[FI] * mesh.magSf()[faceI];
        }
        forAll(F_out, FI)
        {
            (*F_out[FI])[faceI] = flux_out[FI] * mesh.magSf()[faceI];
        }
    }

    // iterate over external faces
    // assume Ql = Qr
    forAll(mesh.boundaryMesh(), patchI)
    {
        const Foam::polyPatch& p = mesh.boundaryMesh()[patchI];
        forAll(p, localFaceI)
        {
            const Foam::label& cellI = p.faceCells()[localFaceI];
            const Foam::vector& n = p.faceNormals()[localFaceI];
            Foam::List<Foam::scalar> Ql(Q.size());
            Foam::List<Foam::scalar> Qauxl(Q.size());
            forAll(Q, QI)
            {
                Ql[QI] = (*Q[QI])[cellI];
            }
            forAll(Qaux, QauxI)
            {
                Qauxl[QauxI] = (*Qaux[QauxI])[cellI];
            }

            const Foam::List<Foam::scalar> flux_in = localNumericalFlux(Ql, Ql, Qauxl, Qauxl, n);
            const Foam::List<Foam::scalar> flux_out = localNumericalFlux(Ql, Ql, Qauxl, Qauxl, n);
            forAll(F_in, FI)
            {
                F_in[FI]->boundaryFieldRef()[patchI][localFaceI] = flux_in[FI]* mesh.magSf()[localFaceI];
            }
            forAll(F_out, FI)
            {
                F_out[FI]->boundaryFieldRef()[patchI][localFaceI] = flux_out[FI]* mesh.magSf()[localFaceI];
            }
        }
    }
}


inline void updateNumericalFlux
(
    Foam::List<Foam::surfaceScalarField*>& F,
    const Foam::List<Foam::volScalarField*>& Q,
    const Foam::List<Foam::volScalarField*>& Qaux
)
{
    const Foam::fvMesh& mesh = Q[0]->mesh();

    // iterate over internal faces
    for (int faceI = 0; faceI < mesh.nInternalFaces(); ++faceI) {
        const Foam::vector n = mesh.Sf()[faceI] / mesh.magSf()[faceI];
        const Foam::label& own = mesh.faceOwner()[faceI];
        const Foam::label& nei = mesh.faceNeighbour()[faceI];

        Foam::List<Foam::scalar> Ql(Q.size()), Qr(Q.size());
        Foam::List<Foam::scalar> Qauxl(Qaux.size()), Qauxr(Qaux.size());
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

        const Foam::List<Foam::scalar> flux = localNumericalFlux(Ql, Qr, Qauxl, Qauxr, n);
        forAll(F, FI)
        {
            (*F[FI])[faceI] = flux[FI] * mesh.magSf()[faceI];
        }
    }

    // iterate over external faces
    // assume Ql = Qr
    forAll(mesh.boundaryMesh(), patchI)
    {
        const Foam::polyPatch& p = mesh.boundaryMesh()[patchI];
        forAll(p, localFaceI)
        {
            const Foam::label& cellI = p.faceCells()[localFaceI];
            const Foam::vector& n = p.faceNormals()[localFaceI];
            Foam::List<Foam::scalar> Ql(Q.size());
            Foam::List<Foam::scalar> Qauxl(Q.size());
            forAll(Q, QI)
            {
                Ql[QI] = (*Q[QI])[cellI];
            }
            forAll(Qaux, QauxI)
            {
                Qauxl[QauxI] = (*Qaux[QauxI])[cellI];
            }

            const Foam::List<Foam::scalar> flux = localNumericalFlux(Ql, Ql, Qauxl, Qauxl, n);
            forAll(F, FI)
            {
                F[FI]->boundaryFieldRef()[patchI][localFaceI] = flux[FI]* mesh.magSf()[localFaceI];
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

    const Foam::scalarField& V = mesh.V();
    const Foam::scalarField& magSf = mesh.magSf();
    const Foam::cellList& cells = mesh.cells();

    Foam::scalarField cellInradius(mesh.nCells(), 0.0);

    forAll(cells, celli)
    {
        const Foam::cell& c = cells[celli];

        Foam::scalar S = 0.0;
        forAll(c, fi)
        {
            const int faceI = c[fi];
            S += magSf[faceI];
        }

        // Simple inradius approximation
        cellInradius[celli] = 3.0 * V[celli] / S;
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
    Foam::surfaceVectorField n = mesh.Sf()/mesh.magSf();

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

    return dt;
}

} //end namespace numerics
