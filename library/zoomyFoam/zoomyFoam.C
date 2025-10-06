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
#include "Model.h"

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
    forAll(Q, QI)
    {
        Q[QI] = new volScalarField (
            IOobject
            (
                "Q" + std::to_string(QI),
                runTime.name(),
                mesh,
                IOobject::MUST_READ,
                IOobject::AUTO_WRITE
            ),
            mesh
        );
    }
    forAll(Qaux, QauxI)
    {
        Qaux[QauxI] = new volScalarField (
            IOobject
            (
                "Qaux" + std::to_string(QauxI),
                runTime.name(),
                mesh,
                IOobject::MUST_READ,
                IOobject::AUTO_WRITE
            ),
            mesh
        );
    }

    List<surfaceScalarField*> F_in(Q.size());
    List<surfaceScalarField*> F_out(Q.size());
    forAll(F_in, FI)
    {
        F_in[FI] = new surfaceScalarField (
            IOobject
            (
                "F_in" + std::to_string(FI),
                runTime.name(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar("", Q[FI]->dimensions() * dimVelocity * dimArea, 0)
        );
    }
    forAll(F_out, FI)
    {
        F_out[FI] = new surfaceScalarField (
            IOobject
            (
                "F_in" + std::to_string(FI),
                runTime.name(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar("", Q[FI]->dimensions() * dimVelocity * dimArea, 0)
        );
    }

    surfaceScalarField minInradius = numerics::computeFaceMinInradius(mesh, runTime);

    const dimensionedScalar diffusivity("diffusivity", dimKinematicViscosity, 0.01);

    scalar Co = readScalar(runTime.controlDict().lookup("maxCo"));
    scalar dt = numerics::compute_dt(Q, Qaux, minInradius, Co);
    numerics::correctBoundaryQ(Q, Qaux, runTime.value());

    while (runTime.loop())
    {
        Info<< nl << "Time = " << runTime.userTimeName() << nl << endl;

        dt = numerics::compute_dt(Q, Qaux, minInradius, Co);
        runTime.setDeltaT(dt);
        
        numerics::updateNumericalQuasilinearFlux(F_in, F_out, Q, Qaux);
        //numerics::updateNumericalFlux(F_in, Q, Qaux);
        forAll(Q, QI)
        {
            fvScalarMatrix
            (
                fvm::ddt(*Q[QI]) + fvc::div(*F_in[QI]) - fvm::laplacian(diffusivity, *Q[QI])
                // fvm::ddt(*Q[QI]) + fvc::div(*F_in[QI]) - fvm::laplacian(diffusivity, *Q[QI])
            ).solve();
        }
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
