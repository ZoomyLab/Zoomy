#include "plotfile_utils.H"

using namespace amrex;


void
WriteSingleLevelPlotfileWithTopo (const std::string& plotfilename,
                                  const MultiFab& mf,
                                  const MultiFab& mf_nd,
                                  const Vector<std::string>& varnames,
                                  const Geometry& my_geom,
                                  Real time,
                                  const int level_steps,
                                  const std::string &versionName,
                                  const std::string &levelPrefix,
                                  const std::string &mfPrefix,
                                  const Vector<std::string>& extra_dirs)
{
    AMREX_ALWAYS_ASSERT(mf.nComp() == varnames.size());

    int level = 0;

    bool callBarrier(false);
    PreBuildDirectorHierarchy(plotfilename, levelPrefix, 1, callBarrier);
    if (!extra_dirs.empty()) {
        for (const auto& d : extra_dirs) {
            const std::string ed = plotfilename+"/"+d;
            PreBuildDirectorHierarchy(ed, levelPrefix, 1, callBarrier);
        }
    }
    ParallelDescriptor::Barrier();

    BoxArray ba = mf.boxArray();
    int nlevels = 1;

    if (ParallelDescriptor::MyProc() == ParallelDescriptor::NProcs()-1) {
        auto f = [=]() {
            VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);
            std::string HeaderFileName(plotfilename + "/Header");
            std::ofstream HeaderFile;
            HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
            HeaderFile.open(HeaderFileName.c_str(), std::ofstream::out   |
                                                    std::ofstream::trunc |
                                                    std::ofstream::binary);
            if( ! HeaderFile.good()) FileOpenFailed(HeaderFileName);
            WriteGenericPlotfileHeaderWithTopo(HeaderFile, nlevels, ba, varnames,
                                               my_geom, time, level_steps, versionName,
                                               levelPrefix, mfPrefix);
        };

        if (AsyncOut::UseAsyncOut()) {
            AsyncOut::Submit(std::move(f));
        } else {
            f();
        }
    }

    std::string mf_nodal_prefix = "Nu_nd";
    if (AsyncOut::UseAsyncOut()) {
        VisMF::AsyncWrite(mf   ,MultiFabFileFullPrefix(level, plotfilename, levelPrefix, mfPrefix), true);
        VisMF::AsyncWrite(mf_nd,MultiFabFileFullPrefix(level, plotfilename, levelPrefix, mf_nodal_prefix), true);
    } else {
        const MultiFab* data;
        std::unique_ptr<MultiFab> mf_tmp;
        if (mf.nGrowVect() != 0) {
            mf_tmp = std::make_unique<MultiFab>(mf.boxArray(),
                                                mf.DistributionMap(),
                                                mf.nComp(), 0, MFInfo(),
                                                mf.Factory());
            MultiFab::Copy(*mf_tmp, mf, 0, 0, mf.nComp(), 0);
            data = mf_tmp.get();
        } else {
            data = &mf;
        }
        VisMF::Write(*data, MultiFabFileFullPrefix(level, plotfilename, levelPrefix, mfPrefix));
        VisMF::Write(mf_nd, MultiFabFileFullPrefix(level, plotfilename, levelPrefix, mf_nodal_prefix));
    }
}

void
WriteGenericPlotfileHeaderWithTopo (std::ostream &HeaderFile,
                                    const int nlevels,
                                    const BoxArray& bArray,
                                    const Vector<std::string>& varnames,
                                    const Geometry& my_geom,
                                    const Real my_time,
                                    const int level_steps,
                                    const std::string &versionName,
                                    const std::string &levelPrefix,
                                    const std::string &mfPrefix)
{
    HeaderFile.precision(17);

    int finest_level = 0;

    // ---- this is the generic plot file type name
    HeaderFile << versionName << '\n';

    HeaderFile << varnames.size() << '\n';

    for (int ivar = 0; ivar < varnames.size(); ++ivar) {
        HeaderFile << varnames[ivar] << "\n";
    }
    HeaderFile << AMREX_SPACEDIM << '\n';
    HeaderFile << my_time << '\n';
    HeaderFile << finest_level << '\n';
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        HeaderFile << my_geom.ProbLo(i) << ' ';
    }
    HeaderFile << '\n';
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        HeaderFile << my_geom.ProbHi(i) << ' ';
    }
    HeaderFile << '\n';
    HeaderFile << " " << ' '; // ref_ratio
    HeaderFile << '\n';
    HeaderFile << my_geom.Domain() << ' ';
    HeaderFile << '\n';
    HeaderFile << level_steps << ' ';
    HeaderFile << '\n';
    for (int k = 0; k < AMREX_SPACEDIM; ++k) {
        HeaderFile << my_geom.CellSize()[k] << ' ';
    }
    HeaderFile << '\n';
    HeaderFile << (int) my_geom.Coord() << '\n';
    HeaderFile << "0\n";

    int level = 0;
    HeaderFile << level << ' ' << bArray.size() << ' ' << my_time << '\n';
    HeaderFile << level_steps << '\n';

    const IntVect& domain_lo = my_geom.Domain().smallEnd();
    for (int i = 0; i < bArray.size(); ++i)
    {
        // Need to shift because the RealBox ctor we call takes the
        // physical location of index (0,0,0).  This does not affect
        // the usual cases where the domain index starts with 0.
        const Box& b = shift(bArray[i], -domain_lo);
        RealBox loc = RealBox(b, my_geom.CellSize(), my_geom.ProbLo());
        for (int n = 0; n < AMREX_SPACEDIM; ++n) {
            HeaderFile << loc.lo(n) << ' ' << loc.hi(n) << '\n';
        }
    }

    HeaderFile << MultiFabHeaderPath(level, levelPrefix, mfPrefix) << '\n';

    HeaderFile << "1" << "\n";
    HeaderFile << "3" << "\n";
    HeaderFile << "amrexvec_nu_x" << "\n";
    HeaderFile << "amrexvec_nu_y" << "\n";
    HeaderFile << "amrexvec_nu_z" << "\n";
    std::string mf_nodal_prefix = "Nu_nd";
    for (int level = 0; level <= finest_level; ++level) {
        HeaderFile << MultiFabHeaderPath(level, levelPrefix, mf_nodal_prefix) << '\n';
    }
}
