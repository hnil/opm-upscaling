dnl -*- autoconf -*-
# Macros needed to find opm-upscaling and dependent libraries.  They are called by
# the macros in ${top_src_dir}/dependencies.m4, which is generated by
# "dunecontrol autogen"

# Additional checks needed to build opm-upscaling
# This macro should be invoked by every module which depends on opm-upscaling, as
# well as by opm-upscaling itself
AC_DEFUN([OPM_UPSCALING_CHECKS])

# Additional checks needed to find opm-upscaling
# This macro should be invoked by every module which depends on opm-upscaling, but
# not by opm-upscaling itself
AC_DEFUN([OPM_UPSCALING_CHECK_MODULE],
[
  OPM_PORSOL_CHECK_MODULES([opm-upscaling],
                     [opm/upscaling/SinglePhaseUpscaler.hpp])
])
