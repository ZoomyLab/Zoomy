/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) YEAR AUTHOR, AFFILIATION
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

Description
    Template for use with codeStream.

\*---------------------------------------------------------------------------*/

#include "dictionary.H"
#include "Ostream.H"
#include "Pstream.H"
#include "pointField.H"
#include "tensor.H"
#include "unitConversion.H"

//{{{ begin codeInclude
#line 24 "/home/ingo/Git/Zoomy/web/tutorials/precice/simple/0/alpha.water/#codeStream"
#include "fvMesh.H"
//}}} end codeInclude

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

//{{{ begin localCode

//}}} end localCode


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

extern "C" void codeStream_9dbe4ef02d04278a12bf38f5908314401f800066(Foam::Ostream& os, const Foam::dictionary& dict)
{
//{{{ begin code
    #line 39 "/home/ingo/Git/Zoomy/web/tutorials/precice/simple/0/alpha.water/#codeStream"
const IOdictionary& d = static_cast<const IOdictionary&>(dict);
                const fvMesh& mesh = refCast<const fvMesh>(d.db());

                const vectorField& cellMidpoints = mesh.cellCentres();
                const scalarField y(cellMidpoints.component(vector::Y));
                const scalarField x(cellMidpoints.component(vector::X));

		scalarField alpha(y.size(), 0);
		forAll(alpha, i)
		{
                    if (x[i] > 0.2 && x[i] < 0.3)
                    {
                        //if (y[i] < 0.04)
                        //{
                        //    alpha[i] = 1;
                        //}
                    }
                    if(y[i] < 0.02)
		    {
		         alpha[i] = 1;
		    }
		}

                alpha.writeEntry("", os);
//}}} end code
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //

