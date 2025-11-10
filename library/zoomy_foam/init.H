#include <string>
#include "argList.H"
#include "dimensionSets.H"
#include "dimensionedScalar.H"
#include "messageStream.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "List.H"
#include "zeroGradientFvPatchFields.H"
#include "fixedValueFvPatchFields.H"
#include "emptyFvPatchFields.H"
#include "Model.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void initialize_fields(word runTimeName, const fvMesh& mesh, List<volScalarField*>& Q, List<volScalarField*>& Qaux, List<surfaceScalarField*>& Dp, List<surfaceScalarField*>& Dm)
{

    forAll(Q, QI)
    {
        word name = "Q" + std::to_string(QI);
        Q[QI] = new volScalarField (
            IOobject
            (
                name,
                runTimeName,
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar(name, dimless, scalar(0.0))
        );
    }
    forAll(Q, QI)
    {
        forAll(Q[QI]->boundaryField(), patchI)
        {
            const word& patchName = mesh.boundary()[patchI].name();
            if (findIndex(Model::map_boundary_tag_to_function_index, patchName) != -1)
            {
                Q[QI]->boundaryFieldRef().set
                (
                    patchI,
                    new fixedValueFvPatchScalarField
                    (
                        Q[QI]->boundaryField()[patchI].patch(),
                        Q[QI]->internalField()
                    )
                );
        
                Q[QI]->boundaryFieldRef()[patchI] = 1.0;
            }
            else
            {
                Q[QI]->boundaryFieldRef().set
                (
                    patchI,
                    new emptyFvPatchScalarField
                    (
                        Q[QI]->boundaryField()[patchI].patch(),
                        Q[QI]->internalField()
                    )
                );
            }
        }
    }

    forAll(Qaux, QauxI)
    {
        word name = "Qaux" + std::to_string(QauxI);
        Qaux[QauxI] = new volScalarField (
            IOobject
            (
                name,
                runTimeName,
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar(name, dimless, scalar(0.0))
        );
    }
    forAll(Qaux, QauxI)
    {
        forAll(Qaux[QauxI]->boundaryField(), patchI)
        {
            const word& patchName = mesh.boundary()[patchI].name();
            if (findIndex(Model::map_boundary_tag_to_function_index, patchName) != -1)
            {
                Qaux[QauxI]->boundaryFieldRef().set
                (
                    patchI,
                    new zeroGradientFvPatchScalarField
                    (
                        Qaux[QauxI]->boundaryField()[patchI].patch(),
                        Qaux[QauxI]->internalField()
                    )
                );
            }
            else
            {
                Qaux[QauxI]->boundaryFieldRef().set
                (
                    patchI,
                    new emptyFvPatchScalarField
                    (
                        Qaux[QauxI]->boundaryField()[patchI].patch(),
                        Qaux[QauxI]->internalField()
                    )
                );
            }
        }
    }
    forAll(Dp, DpI)
    {
        Dp[DpI] = new surfaceScalarField (
            IOobject
            (
                "Dp" + std::to_string(DpI),
                runTimeName,
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar("", Q[DpI]->dimensions() * dimVelocity * dimArea, 0)
        );
    }
    forAll(Dm, DmI)
    {
        Dm[DmI] = new surfaceScalarField (
            IOobject
            (
                "Dm" + std::to_string(DmI),
                runTimeName,
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar("", Q[DmI]->dimensions() * dimVelocity * dimArea, 0)
        );
    }
}
