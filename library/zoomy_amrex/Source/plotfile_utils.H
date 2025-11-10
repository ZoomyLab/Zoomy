#include "AMReX_PlotFileUtil.H"

void
WriteSingleLevelPlotfileWithTopo (const std::string& plotfilename,
                                  const amrex::MultiFab& mf,
                                  const amrex::MultiFab& mf_nd,
                                  const amrex::Vector<std::string>& varnames,
                                  const amrex::Geometry& my_geom,
                                  amrex::Real time,
                                  const int level_steps,
                                  const std::string &versionName = "HyperCLaw-V1.1",
                                  const std::string &levelPrefix = "Level_",
                                  const std::string &mfPrefix = "Cell",
                                  const amrex::Vector<std::string>& extra_dirs = amrex::Vector<std::string>());

void
WriteGenericPlotfileHeaderWithTopo (std::ostream &HeaderFile,
                                    const int nlevels,
                                    const amrex::BoxArray& bArray,
                                    const amrex::Vector<std::string>& varnames,
                                    const amrex::Geometry& my_geom,
                                    const amrex::Real my_time,
                                    const int level_steps,
                                    const std::string &versionName,
                                    const std::string &levelPrefix,
                                    const std::string &mfPrefix);
