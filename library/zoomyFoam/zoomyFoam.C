/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2025 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    shallowTest

Description

\*---------------------------------------------------------------------------*/

#include <string>
#include "UList.H"
#include "argList.H"
#include "dimensionSets.H"
#include "dimensionedScalar.H"
#include "fvcDiv.H"
#include "fvmDdt.H"
#include "fvmLaplacian.H"
#include "messageStream.H"
#include "vector.H"
#include "vectorField.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "List.H"
#include "numerics.H"
#include "zeroGradientFvPatchFields.H"
#include "fixedValueFvPatchFields.H"
#include "emptyFvPatchFields.H"
#include "Model.h"
#include "init.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList args(argc, argv);
    if (!args.checkRootCase())
    {
        FatalError.exit();
    }

    Info<< "Create time\n" << endl;
    Time runTime(Foam::Time::controlDictName, args);

    Info
        << "Create mesh for time = "
        << runTime.name() << Foam::nl << Foam::endl;


    fvMesh mesh
    (
        IOobject
        (
            fvMesh::defaultRegion,
            runTime.name(),
            runTime,
            IOobject::MUST_READ
        )
    );


    List<volScalarField*> Q (Model::n_dof_q);
    List<volScalarField*> Qaux (Model::n_dof_qaux);
    List<surfaceScalarField*> F(Q.size());
    initialize_fields(runTime.name(), mesh, Q, Qaux, F);

    // Set initial condition based on position
    forAll(Q[0]->internalField(), cellI)
    {
        const point& C = mesh.C()[cellI];   // cell center
        scalar x = C.x();
        Q[0]->internalFieldRef()[cellI] = 1.35;
        if (x > 5) Q[0]->internalFieldRef()[cellI] = 1.4;
    }   
    forAll(Q, QI)
    {
        Q[QI]->write();
    }
    forAll(Qaux, QauxI)
    {
        Qaux[QauxI]->write();
    }
    forAll(F, FI)
    {
        F[FI]->write();
    }

    surfaceScalarField minInradius = numerics::computeFaceMinInradius(mesh, runTime);

    const dimensionedScalar diffusivity("diffusivity", dimKinematicViscosity, 0.01);

    scalar Co = readScalar(runTime.controlDict().lookup("maxCo"));
    scalar dt = numerics::compute_dt(Q, Qaux, minInradius, Co);
    numerics::correctBoundaryQ(Q, Qaux, runTime.value());


    while (runTime.loop())
    {
        Info<< nl << "Time = " << runTime.userTimeName() << nl << endl;

        Info << "dt" << endl;
        dt = numerics::compute_dt(Q, Qaux, minInradius, Co);
        runTime.setDeltaT(dt);

        Info << "dt= " << dt << endl;
        
        Info << "flux" << endl;
        numerics::updateNumericalQuasilinearFlux(F, Q, Qaux);
        Info << "solve" << endl;
        forAll(Q, QI)
        {
            fvScalarMatrix
            (
                //fvm::ddt(*Q[QI]) + fvc::div(*F[QI]) - fvm::laplacian(diffusivity, *Q[QI])
                fvm::ddt(*Q[QI]) + fvc::div(*F[QI])
                // fvm::ddt(*Q[QI]) + fvc::div(*F[QI]) - fvm::laplacian(diffusivity, *Q[QI])
            ).solve();
        }
        //const scalarField& V = mesh.V();
        //forAll(Q, QI)
        //{
        //    volScalarField divFlux = fvc::div(*F[QI]);
        //    forAll((*Q[QI]), i)
        //    {
        //        (*Q[QI])[i] -= dt * divFlux[i] / V[i] ;
        //    }
        //}
        Info << "bc" << endl;
        numerics::correctBoundaryQ(Q, Qaux, runTime.value());
        runTime.write();
    }

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< nl << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
        << "  ClockTime = " << runTime.elapsedClockTime() << " s"
        << nl << endl;

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
