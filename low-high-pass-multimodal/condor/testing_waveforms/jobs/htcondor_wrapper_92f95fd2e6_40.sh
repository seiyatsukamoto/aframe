#!/usr/bin/env bash

# Wrapper script that is to be configured as htcondor's main executable file

htcondor_wrapper() {
    # helper to select the correct python executable
    _law_python() {
        command -v python &> /dev/null && python "$@" || python3 "$@"
    }

    #
    # detect variables
    #

    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_file_base="$( basename "${this_file}" )"

    # get the job number
    export LAW_HTCONDOR_JOB_NUMBER="${LAW_HTCONDOR_JOB_PROCESS}"
    if [ -z "${LAW_HTCONDOR_JOB_NUMBER}" ]; then
        >&2 echo "could not determine htcondor job number"
        return "1"
    fi
    # htcondor process numbers start at 0, law job numbers at 1, so increment
    ((LAW_HTCONDOR_JOB_NUMBER++))
    echo "running ${this_file_base} for job number ${LAW_HTCONDOR_JOB_NUMBER}"


    #
    # job argument definitons, depending on LAW_HTCONDOR_JOB_NUMBER
    #

    # definition
    local htcondor_job_arguments_map
    declare -A htcondor_job_arguments_map
    htcondor_job_arguments_map=(
        ['1']="aframe.tasks.data.waveforms.testing DeployTestingWaveforms LS1sb2ctZmlsZT1OT19TVFIgLS1jbGVhci1sb2dzPUZhbHNlIC0tZGV2PVRydWUgLS1ncHVzPTAsMiwzIC0taW1hZ2U9L2hvbWUvc2VpeWEudHN1a2Ftb3RvL2FmcmFtZS9pbWFnZXMvZGF0YS5zaWYgLS1jb250YWluZXItcm9vdD0vaG9tZS9zZWl5YS50c3VrYW1vdG8vYWZyYW1lL2ltYWdlcy8gLS1qb2ItbG9nPScnIC0tYWNjb3VudGluZy1ncm91cC11c2VyPXNlaXlhLnRzdWthbW90byAtLWFjY291bnRpbmctZ3JvdXA9bGlnby5kZXYubzQuY2JjLmFsbHNreS5hZnJhbWUgLS1yZXF1ZXN0LWRpc2s9MjAwTUIgLS1yZXF1ZXN0LW1lbW9yeT0xNkdCIC0tcmVxdWVzdC1jcHVzPTEgLS1udW0tc2lnbmFscz01MDAwMDAgLS1zYW1wbGUtcmF0ZT0yMDQ4LjAgLS13YXZlZm9ybS1kdXJhdGlvbj04LjAgLS1wcmlvcj1wcmlvcnMucHJpb3JzLmVuZF9vM19yYXRlc2FuZHBvcHMgLS1taW5pbXVtLWZyZXF1ZW5jeT0yMC4wIC0tcmVmZXJlbmNlLWZyZXF1ZW5jeT01MC4wIC0td2F2ZWZvcm0tYXBwcm94aW1hbnQ9SU1SUGhlbm9tWFBITSAtLWNvYWxlc2NlbmNlLXRpbWU9NC4wIC0tc3RhcnQ9MTI0MTQ0Mzc4My4wIC0tZW5kPTEyNDQwMzU3ODMuMCAtLWlmb3M9J1siSDEiLCAiTDEiXScgLS1zaGlmdHM9J1swLCAxXScgLS1zcGFjaW5nPTE2LjAgLS1idWZmZXI9MTYuMCAtLWhpZ2hwYXNzPTMyLjAgLS1sb3dwYXNzPScnIC0tc25yLXRocmVzaG9sZD00LjAgLS1wc2QtbGVuZ3RoPTY0LjAgLS1zZWVkPTExMjIgLS1vdXRwdXQtZGlyPS9ob21lL3NlaXlhLnRzdWthbW90by9hZnJhbWUvbG93LWhpZ2gtcGFzcy1tdWx0aW1vZGFsL2RhdGEvdGVzdCAtLWNvbmRvci1kaXJlY3Rvcnk9L2hvbWUvc2VpeWEudHN1a2Ftb3RvL2FmcmFtZS9sb3ctaGlnaC1wYXNzLW11bHRpbW9kYWwvY29uZG9yL3Rlc3Rpbmdfd2F2ZWZvcm1zIC0tbG9jYWwtc2NoZWR1bGVyPVRydWU= MTEyNg== 1 no LQ=="
        ['2']="aframe.tasks.data.waveforms.testing DeployTestingWaveforms LS1sb2ctZmlsZT1OT19TVFIgLS1jbGVhci1sb2dzPUZhbHNlIC0tZGV2PVRydWUgLS1ncHVzPTAsMiwzIC0taW1hZ2U9L2hvbWUvc2VpeWEudHN1a2Ftb3RvL2FmcmFtZS9pbWFnZXMvZGF0YS5zaWYgLS1jb250YWluZXItcm9vdD0vaG9tZS9zZWl5YS50c3VrYW1vdG8vYWZyYW1lL2ltYWdlcy8gLS1qb2ItbG9nPScnIC0tYWNjb3VudGluZy1ncm91cC11c2VyPXNlaXlhLnRzdWthbW90byAtLWFjY291bnRpbmctZ3JvdXA9bGlnby5kZXYubzQuY2JjLmFsbHNreS5hZnJhbWUgLS1yZXF1ZXN0LWRpc2s9MjAwTUIgLS1yZXF1ZXN0LW1lbW9yeT0xNkdCIC0tcmVxdWVzdC1jcHVzPTEgLS1udW0tc2lnbmFscz01MDAwMDAgLS1zYW1wbGUtcmF0ZT0yMDQ4LjAgLS13YXZlZm9ybS1kdXJhdGlvbj04LjAgLS1wcmlvcj1wcmlvcnMucHJpb3JzLmVuZF9vM19yYXRlc2FuZHBvcHMgLS1taW5pbXVtLWZyZXF1ZW5jeT0yMC4wIC0tcmVmZXJlbmNlLWZyZXF1ZW5jeT01MC4wIC0td2F2ZWZvcm0tYXBwcm94aW1hbnQ9SU1SUGhlbm9tWFBITSAtLWNvYWxlc2NlbmNlLXRpbWU9NC4wIC0tc3RhcnQ9MTI0MTQ0Mzc4My4wIC0tZW5kPTEyNDQwMzU3ODMuMCAtLWlmb3M9J1siSDEiLCAiTDEiXScgLS1zaGlmdHM9J1swLCAxXScgLS1zcGFjaW5nPTE2LjAgLS1idWZmZXI9MTYuMCAtLWhpZ2hwYXNzPTMyLjAgLS1sb3dwYXNzPScnIC0tc25yLXRocmVzaG9sZD00LjAgLS1wc2QtbGVuZ3RoPTY0LjAgLS1zZWVkPTExMjIgLS1vdXRwdXQtZGlyPS9ob21lL3NlaXlhLnRzdWthbW90by9hZnJhbWUvbG93LWhpZ2gtcGFzcy1tdWx0aW1vZGFsL2RhdGEvdGVzdCAtLWNvbmRvci1kaXJlY3Rvcnk9L2hvbWUvc2VpeWEudHN1a2Ftb3RvL2FmcmFtZS9sb3ctaGlnaC1wYXNzLW11bHRpbW9kYWwvY29uZG9yL3Rlc3Rpbmdfd2F2ZWZvcm1zIC0tbG9jYWwtc2NoZWR1bGVyPVRydWU= MTEyOQ== 1 no LQ=="
        ['3']="aframe.tasks.data.waveforms.testing DeployTestingWaveforms LS1sb2ctZmlsZT1OT19TVFIgLS1jbGVhci1sb2dzPUZhbHNlIC0tZGV2PVRydWUgLS1ncHVzPTAsMiwzIC0taW1hZ2U9L2hvbWUvc2VpeWEudHN1a2Ftb3RvL2FmcmFtZS9pbWFnZXMvZGF0YS5zaWYgLS1jb250YWluZXItcm9vdD0vaG9tZS9zZWl5YS50c3VrYW1vdG8vYWZyYW1lL2ltYWdlcy8gLS1qb2ItbG9nPScnIC0tYWNjb3VudGluZy1ncm91cC11c2VyPXNlaXlhLnRzdWthbW90byAtLWFjY291bnRpbmctZ3JvdXA9bGlnby5kZXYubzQuY2JjLmFsbHNreS5hZnJhbWUgLS1yZXF1ZXN0LWRpc2s9MjAwTUIgLS1yZXF1ZXN0LW1lbW9yeT0xNkdCIC0tcmVxdWVzdC1jcHVzPTEgLS1udW0tc2lnbmFscz01MDAwMDAgLS1zYW1wbGUtcmF0ZT0yMDQ4LjAgLS13YXZlZm9ybS1kdXJhdGlvbj04LjAgLS1wcmlvcj1wcmlvcnMucHJpb3JzLmVuZF9vM19yYXRlc2FuZHBvcHMgLS1taW5pbXVtLWZyZXF1ZW5jeT0yMC4wIC0tcmVmZXJlbmNlLWZyZXF1ZW5jeT01MC4wIC0td2F2ZWZvcm0tYXBwcm94aW1hbnQ9SU1SUGhlbm9tWFBITSAtLWNvYWxlc2NlbmNlLXRpbWU9NC4wIC0tc3RhcnQ9MTI0MTQ0Mzc4My4wIC0tZW5kPTEyNDQwMzU3ODMuMCAtLWlmb3M9J1siSDEiLCAiTDEiXScgLS1zaGlmdHM9J1swLCAxXScgLS1zcGFjaW5nPTE2LjAgLS1idWZmZXI9MTYuMCAtLWhpZ2hwYXNzPTMyLjAgLS1sb3dwYXNzPScnIC0tc25yLXRocmVzaG9sZD00LjAgLS1wc2QtbGVuZ3RoPTY0LjAgLS1zZWVkPTExMjIgLS1vdXRwdXQtZGlyPS9ob21lL3NlaXlhLnRzdWthbW90by9hZnJhbWUvbG93LWhpZ2gtcGFzcy1tdWx0aW1vZGFsL2RhdGEvdGVzdCAtLWNvbmRvci1kaXJlY3Rvcnk9L2hvbWUvc2VpeWEudHN1a2Ftb3RvL2FmcmFtZS9sb3ctaGlnaC1wYXNzLW11bHRpbW9kYWwvY29uZG9yL3Rlc3Rpbmdfd2F2ZWZvcm1zIC0tbG9jYWwtc2NoZWR1bGVyPVRydWU= MTEzMA== 1 no LQ=="
        ['4']="aframe.tasks.data.waveforms.testing DeployTestingWaveforms LS1sb2ctZmlsZT1OT19TVFIgLS1jbGVhci1sb2dzPUZhbHNlIC0tZGV2PVRydWUgLS1ncHVzPTAsMiwzIC0taW1hZ2U9L2hvbWUvc2VpeWEudHN1a2Ftb3RvL2FmcmFtZS9pbWFnZXMvZGF0YS5zaWYgLS1jb250YWluZXItcm9vdD0vaG9tZS9zZWl5YS50c3VrYW1vdG8vYWZyYW1lL2ltYWdlcy8gLS1qb2ItbG9nPScnIC0tYWNjb3VudGluZy1ncm91cC11c2VyPXNlaXlhLnRzdWthbW90byAtLWFjY291bnRpbmctZ3JvdXA9bGlnby5kZXYubzQuY2JjLmFsbHNreS5hZnJhbWUgLS1yZXF1ZXN0LWRpc2s9MjAwTUIgLS1yZXF1ZXN0LW1lbW9yeT0xNkdCIC0tcmVxdWVzdC1jcHVzPTEgLS1udW0tc2lnbmFscz01MDAwMDAgLS1zYW1wbGUtcmF0ZT0yMDQ4LjAgLS13YXZlZm9ybS1kdXJhdGlvbj04LjAgLS1wcmlvcj1wcmlvcnMucHJpb3JzLmVuZF9vM19yYXRlc2FuZHBvcHMgLS1taW5pbXVtLWZyZXF1ZW5jeT0yMC4wIC0tcmVmZXJlbmNlLWZyZXF1ZW5jeT01MC4wIC0td2F2ZWZvcm0tYXBwcm94aW1hbnQ9SU1SUGhlbm9tWFBITSAtLWNvYWxlc2NlbmNlLXRpbWU9NC4wIC0tc3RhcnQ9MTI0MTQ0Mzc4My4wIC0tZW5kPTEyNDQwMzU3ODMuMCAtLWlmb3M9J1siSDEiLCAiTDEiXScgLS1zaGlmdHM9J1swLCAxXScgLS1zcGFjaW5nPTE2LjAgLS1idWZmZXI9MTYuMCAtLWhpZ2hwYXNzPTMyLjAgLS1sb3dwYXNzPScnIC0tc25yLXRocmVzaG9sZD00LjAgLS1wc2QtbGVuZ3RoPTY0LjAgLS1zZWVkPTExMjIgLS1vdXRwdXQtZGlyPS9ob21lL3NlaXlhLnRzdWthbW90by9hZnJhbWUvbG93LWhpZ2gtcGFzcy1tdWx0aW1vZGFsL2RhdGEvdGVzdCAtLWNvbmRvci1kaXJlY3Rvcnk9L2hvbWUvc2VpeWEudHN1a2Ftb3RvL2FmcmFtZS9sb3ctaGlnaC1wYXNzLW11bHRpbW9kYWwvY29uZG9yL3Rlc3Rpbmdfd2F2ZWZvcm1zIC0tbG9jYWwtc2NoZWR1bGVyPVRydWU= MTEzMQ== 1 no LQ=="
        ['5']="aframe.tasks.data.waveforms.testing DeployTestingWaveforms LS1sb2ctZmlsZT1OT19TVFIgLS1jbGVhci1sb2dzPUZhbHNlIC0tZGV2PVRydWUgLS1ncHVzPTAsMiwzIC0taW1hZ2U9L2hvbWUvc2VpeWEudHN1a2Ftb3RvL2FmcmFtZS9pbWFnZXMvZGF0YS5zaWYgLS1jb250YWluZXItcm9vdD0vaG9tZS9zZWl5YS50c3VrYW1vdG8vYWZyYW1lL2ltYWdlcy8gLS1qb2ItbG9nPScnIC0tYWNjb3VudGluZy1ncm91cC11c2VyPXNlaXlhLnRzdWthbW90byAtLWFjY291bnRpbmctZ3JvdXA9bGlnby5kZXYubzQuY2JjLmFsbHNreS5hZnJhbWUgLS1yZXF1ZXN0LWRpc2s9MjAwTUIgLS1yZXF1ZXN0LW1lbW9yeT0xNkdCIC0tcmVxdWVzdC1jcHVzPTEgLS1udW0tc2lnbmFscz01MDAwMDAgLS1zYW1wbGUtcmF0ZT0yMDQ4LjAgLS13YXZlZm9ybS1kdXJhdGlvbj04LjAgLS1wcmlvcj1wcmlvcnMucHJpb3JzLmVuZF9vM19yYXRlc2FuZHBvcHMgLS1taW5pbXVtLWZyZXF1ZW5jeT0yMC4wIC0tcmVmZXJlbmNlLWZyZXF1ZW5jeT01MC4wIC0td2F2ZWZvcm0tYXBwcm94aW1hbnQ9SU1SUGhlbm9tWFBITSAtLWNvYWxlc2NlbmNlLXRpbWU9NC4wIC0tc3RhcnQ9MTI0MTQ0Mzc4My4wIC0tZW5kPTEyNDQwMzU3ODMuMCAtLWlmb3M9J1siSDEiLCAiTDEiXScgLS1zaGlmdHM9J1swLCAxXScgLS1zcGFjaW5nPTE2LjAgLS1idWZmZXI9MTYuMCAtLWhpZ2hwYXNzPTMyLjAgLS1sb3dwYXNzPScnIC0tc25yLXRocmVzaG9sZD00LjAgLS1wc2QtbGVuZ3RoPTY0LjAgLS1zZWVkPTExMjIgLS1vdXRwdXQtZGlyPS9ob21lL3NlaXlhLnRzdWthbW90by9hZnJhbWUvbG93LWhpZ2gtcGFzcy1tdWx0aW1vZGFsL2RhdGEvdGVzdCAtLWNvbmRvci1kaXJlY3Rvcnk9L2hvbWUvc2VpeWEudHN1a2Ftb3RvL2FmcmFtZS9sb3ctaGlnaC1wYXNzLW11bHRpbW9kYWwvY29uZG9yL3Rlc3Rpbmdfd2F2ZWZvcm1zIC0tbG9jYWwtc2NoZWR1bGVyPVRydWU= MTEzMg== 1 no LQ=="
    )

    # pick
    local htcondor_job_arguments="${htcondor_job_arguments_map[${LAW_HTCONDOR_JOB_NUMBER}]}"
    if [ -z "${htcondor_job_arguments}" ]; then
        >&2 echo "empty htcondor job arguments for LAW_HTCONDOR_JOB_NUMBER ${LAW_HTCONDOR_JOB_NUMBER}"
        return "3"
    fi


    #
    # variable rendering
    #

    # check variables
    local render_variables="eyJqb2JfZmlsZSI6ICJsYXdfam9iXzBlNWMxZDdkOWEuc2giLCAiZXhlY3V0YWJsZV9maWxlIjogImh0Y29uZG9yX3dyYXBwZXJfOTJmOTVmZDJlNl80MC5zaCIsICJpbnB1dF9maWxlcyI6ICJsYXdfam9iXzBlNWMxZDdkOWEuc2ggaHRjb25kb3Jfd3JhcHBlcl85MmY5NWZkMmU2XzQwLnNoIiwgImlucHV0X2ZpbGVzX3JlbmRlciI6ICJsYXdfam9iXzBlNWMxZDdkOWEuc2giLCAiaHRjb25kb3Jfam9iX2FyZ3VtZW50c19tYXAiOiAiWycxJ109XCJhZnJhbWUudGFza3MuZGF0YS53YXZlZm9ybXMudGVzdGluZyBEZXBsb3lUZXN0aW5nV2F2ZWZvcm1zIExTMXNiMmN0Wm1sc1pUMU9UMTlUVkZJZ0xTMWpiR1ZoY2kxc2IyZHpQVVpoYkhObElDMHRaR1YyUFZSeWRXVWdMUzFuY0hWelBUQXNNaXd6SUMwdGFXMWhaMlU5TDJodmJXVXZjMlZwZVdFdWRITjFhMkZ0YjNSdkwyRm1jbUZ0WlM5cGJXRm5aWE12WkdGMFlTNXphV1lnTFMxamIyNTBZV2x1WlhJdGNtOXZkRDB2YUc5dFpTOXpaV2w1WVM1MGMzVnJZVzF2ZEc4dllXWnlZVzFsTDJsdFlXZGxjeThnTFMxcWIySXRiRzluUFNjbklDMHRZV05qYjNWdWRHbHVaeTFuY205MWNDMTFjMlZ5UFhObGFYbGhMblJ6ZFd0aGJXOTBieUF0TFdGalkyOTFiblJwYm1jdFozSnZkWEE5YkdsbmJ5NWtaWFl1YnpRdVkySmpMbUZzYkhOcmVTNWhabkpoYldVZ0xTMXlaWEYxWlhOMExXUnBjMnM5TWpBd1RVSWdMUzF5WlhGMVpYTjBMVzFsYlc5eWVUMHhOa2RDSUMwdGNtVnhkV1Z6ZEMxamNIVnpQVEVnTFMxdWRXMHRjMmxuYm1Gc2N6MDFNREF3TURBZ0xTMXpZVzF3YkdVdGNtRjBaVDB5TURRNExqQWdMUzEzWVhabFptOXliUzFrZFhKaGRHbHZiajA0TGpBZ0xTMXdjbWx2Y2oxd2NtbHZjbk11Y0hKcGIzSnpMbVZ1WkY5dk0xOXlZWFJsYzJGdVpIQnZjSE1nTFMxdGFXNXBiWFZ0TFdaeVpYRjFaVzVqZVQweU1DNHdJQzB0Y21WbVpYSmxibU5sTFdaeVpYRjFaVzVqZVQwMU1DNHdJQzB0ZDJGMlpXWnZjbTB0WVhCd2NtOTRhVzFoYm5ROVNVMVNVR2hsYm05dFdGQklUU0F0TFdOdllXeGxjMk5sYm1ObExYUnBiV1U5TkM0d0lDMHRjM1JoY25ROU1USTBNVFEwTXpjNE15NHdJQzB0Wlc1a1BURXlORFF3TXpVM09ETXVNQ0F0TFdsbWIzTTlKMXNpU0RFaUxDQWlUREVpWFNjZ0xTMXphR2xtZEhNOUoxc3dMQ0F4WFNjZ0xTMXpjR0ZqYVc1blBURTJMakFnTFMxaWRXWm1aWEk5TVRZdU1DQXRMV2hwWjJod1lYTnpQVE15TGpBZ0xTMXNiM2R3WVhOelBTY25JQzB0YzI1eUxYUm9jbVZ6YUc5c1pEMDBMakFnTFMxd2MyUXRiR1Z1WjNSb1BUWTBMakFnTFMxelpXVmtQVEV4TWpJZ0xTMXZkWFJ3ZFhRdFpHbHlQUzlvYjIxbEwzTmxhWGxoTG5SemRXdGhiVzkwYnk5aFpuSmhiV1V2Ykc5M0xXaHBaMmd0Y0dGemN5MXRkV3gwYVcxdlpHRnNMMlJoZEdFdmRHVnpkQ0F0TFdOdmJtUnZjaTFrYVhKbFkzUnZjbms5TDJodmJXVXZjMlZwZVdFdWRITjFhMkZ0YjNSdkwyRm1jbUZ0WlM5c2IzY3RhR2xuYUMxd1lYTnpMVzExYkhScGJXOWtZV3d2WTI5dVpHOXlMM1JsYzNScGJtZGZkMkYyWldadmNtMXpJQzB0Ykc5allXd3RjMk5vWldSMWJHVnlQVlJ5ZFdVPSBNVEV5Tmc9PSAxIG5vIExRPT1cIlxuICAgICAgICBbJzInXT1cImFmcmFtZS50YXNrcy5kYXRhLndhdmVmb3Jtcy50ZXN0aW5nIERlcGxveVRlc3RpbmdXYXZlZm9ybXMgTFMxc2IyY3RabWxzWlQxT1QxOVRWRklnTFMxamJHVmhjaTFzYjJkelBVWmhiSE5sSUMwdFpHVjJQVlJ5ZFdVZ0xTMW5jSFZ6UFRBc01pd3pJQzB0YVcxaFoyVTlMMmh2YldVdmMyVnBlV0V1ZEhOMWEyRnRiM1J2TDJGbWNtRnRaUzlwYldGblpYTXZaR0YwWVM1emFXWWdMUzFqYjI1MFlXbHVaWEl0Y205dmREMHZhRzl0WlM5elpXbDVZUzUwYzNWcllXMXZkRzh2WVdaeVlXMWxMMmx0WVdkbGN5OGdMUzFxYjJJdGJHOW5QU2NuSUMwdFlXTmpiM1Z1ZEdsdVp5MW5jbTkxY0MxMWMyVnlQWE5sYVhsaExuUnpkV3RoYlc5MGJ5QXRMV0ZqWTI5MWJuUnBibWN0WjNKdmRYQTliR2xuYnk1a1pYWXVielF1WTJKakxtRnNiSE5yZVM1aFpuSmhiV1VnTFMxeVpYRjFaWE4wTFdScGMyczlNakF3VFVJZ0xTMXlaWEYxWlhOMExXMWxiVzl5ZVQweE5rZENJQzB0Y21WeGRXVnpkQzFqY0hWelBURWdMUzF1ZFcwdGMybG5ibUZzY3owMU1EQXdNREFnTFMxellXMXdiR1V0Y21GMFpUMHlNRFE0TGpBZ0xTMTNZWFpsWm05eWJTMWtkWEpoZEdsdmJqMDRMakFnTFMxd2NtbHZjajF3Y21sdmNuTXVjSEpwYjNKekxtVnVaRjl2TTE5eVlYUmxjMkZ1WkhCdmNITWdMUzF0YVc1cGJYVnRMV1p5WlhGMVpXNWplVDB5TUM0d0lDMHRjbVZtWlhKbGJtTmxMV1p5WlhGMVpXNWplVDAxTUM0d0lDMHRkMkYyWldadmNtMHRZWEJ3Y205NGFXMWhiblE5U1UxU1VHaGxibTl0V0ZCSVRTQXRMV052WVd4bGMyTmxibU5sTFhScGJXVTlOQzR3SUMwdGMzUmhjblE5TVRJME1UUTBNemM0TXk0d0lDMHRaVzVrUFRFeU5EUXdNelUzT0RNdU1DQXRMV2xtYjNNOUoxc2lTREVpTENBaVRERWlYU2NnTFMxemFHbG1kSE05SjFzd0xDQXhYU2NnTFMxemNHRmphVzVuUFRFMkxqQWdMUzFpZFdabVpYSTlNVFl1TUNBdExXaHBaMmh3WVhOelBUTXlMakFnTFMxc2IzZHdZWE56UFNjbklDMHRjMjV5TFhSb2NtVnphRzlzWkQwMExqQWdMUzF3YzJRdGJHVnVaM1JvUFRZMExqQWdMUzF6WldWa1BURXhNaklnTFMxdmRYUndkWFF0WkdseVBTOW9iMjFsTDNObGFYbGhMblJ6ZFd0aGJXOTBieTloWm5KaGJXVXZiRzkzTFdocFoyZ3RjR0Z6Y3kxdGRXeDBhVzF2WkdGc0wyUmhkR0V2ZEdWemRDQXRMV052Ym1SdmNpMWthWEpsWTNSdmNuazlMMmh2YldVdmMyVnBlV0V1ZEhOMWEyRnRiM1J2TDJGbWNtRnRaUzlzYjNjdGFHbG5hQzF3WVhOekxXMTFiSFJwYlc5a1lXd3ZZMjl1Wkc5eUwzUmxjM1JwYm1kZmQyRjJaV1p2Y20xeklDMHRiRzlqWVd3dGMyTm9aV1IxYkdWeVBWUnlkV1U9IE1URXlPUT09IDEgbm8gTFE9PVwiXG4gICAgICAgIFsnMyddPVwiYWZyYW1lLnRhc2tzLmRhdGEud2F2ZWZvcm1zLnRlc3RpbmcgRGVwbG95VGVzdGluZ1dhdmVmb3JtcyBMUzFzYjJjdFptbHNaVDFPVDE5VFZGSWdMUzFqYkdWaGNpMXNiMmR6UFVaaGJITmxJQzB0WkdWMlBWUnlkV1VnTFMxbmNIVnpQVEFzTWl3eklDMHRhVzFoWjJVOUwyaHZiV1V2YzJWcGVXRXVkSE4xYTJGdGIzUnZMMkZtY21GdFpTOXBiV0ZuWlhNdlpHRjBZUzV6YVdZZ0xTMWpiMjUwWVdsdVpYSXRjbTl2ZEQwdmFHOXRaUzl6WldsNVlTNTBjM1ZyWVcxdmRHOHZZV1p5WVcxbEwybHRZV2RsY3k4Z0xTMXFiMkl0Ykc5blBTY25JQzB0WVdOamIzVnVkR2x1WnkxbmNtOTFjQzExYzJWeVBYTmxhWGxoTG5SemRXdGhiVzkwYnlBdExXRmpZMjkxYm5ScGJtY3RaM0p2ZFhBOWJHbG5ieTVrWlhZdWJ6UXVZMkpqTG1Gc2JITnJlUzVoWm5KaGJXVWdMUzF5WlhGMVpYTjBMV1JwYzJzOU1qQXdUVUlnTFMxeVpYRjFaWE4wTFcxbGJXOXllVDB4TmtkQ0lDMHRjbVZ4ZFdWemRDMWpjSFZ6UFRFZ0xTMXVkVzB0YzJsbmJtRnNjejAxTURBd01EQWdMUzF6WVcxd2JHVXRjbUYwWlQweU1EUTRMakFnTFMxM1lYWmxabTl5YlMxa2RYSmhkR2x2YmowNExqQWdMUzF3Y21sdmNqMXdjbWx2Y25NdWNISnBiM0p6TG1WdVpGOXZNMTl5WVhSbGMyRnVaSEJ2Y0hNZ0xTMXRhVzVwYlhWdExXWnlaWEYxWlc1amVUMHlNQzR3SUMwdGNtVm1aWEpsYm1ObExXWnlaWEYxWlc1amVUMDFNQzR3SUMwdGQyRjJaV1p2Y20wdFlYQndjbTk0YVcxaGJuUTlTVTFTVUdobGJtOXRXRkJJVFNBdExXTnZZV3hsYzJObGJtTmxMWFJwYldVOU5DNHdJQzB0YzNSaGNuUTlNVEkwTVRRME16YzRNeTR3SUMwdFpXNWtQVEV5TkRRd016VTNPRE11TUNBdExXbG1iM005SjFzaVNERWlMQ0FpVERFaVhTY2dMUzF6YUdsbWRITTlKMXN3TENBeFhTY2dMUzF6Y0dGamFXNW5QVEUyTGpBZ0xTMWlkV1ptWlhJOU1UWXVNQ0F0TFdocFoyaHdZWE56UFRNeUxqQWdMUzFzYjNkd1lYTnpQU2NuSUMwdGMyNXlMWFJvY21WemFHOXNaRDAwTGpBZ0xTMXdjMlF0YkdWdVozUm9QVFkwTGpBZ0xTMXpaV1ZrUFRFeE1qSWdMUzF2ZFhSd2RYUXRaR2x5UFM5b2IyMWxMM05sYVhsaExuUnpkV3RoYlc5MGJ5OWhabkpoYldVdmJHOTNMV2hwWjJndGNHRnpjeTF0ZFd4MGFXMXZaR0ZzTDJSaGRHRXZkR1Z6ZENBdExXTnZibVJ2Y2kxa2FYSmxZM1J2Y25rOUwyaHZiV1V2YzJWcGVXRXVkSE4xYTJGdGIzUnZMMkZtY21GdFpTOXNiM2N0YUdsbmFDMXdZWE56TFcxMWJIUnBiVzlrWVd3dlkyOXVaRzl5TDNSbGMzUnBibWRmZDJGMlpXWnZjbTF6SUMwdGJHOWpZV3d0YzJOb1pXUjFiR1Z5UFZSeWRXVT0gTVRFek1BPT0gMSBubyBMUT09XCJcbiAgICAgICAgWyc0J109XCJhZnJhbWUudGFza3MuZGF0YS53YXZlZm9ybXMudGVzdGluZyBEZXBsb3lUZXN0aW5nV2F2ZWZvcm1zIExTMXNiMmN0Wm1sc1pUMU9UMTlUVkZJZ0xTMWpiR1ZoY2kxc2IyZHpQVVpoYkhObElDMHRaR1YyUFZSeWRXVWdMUzFuY0hWelBUQXNNaXd6SUMwdGFXMWhaMlU5TDJodmJXVXZjMlZwZVdFdWRITjFhMkZ0YjNSdkwyRm1jbUZ0WlM5cGJXRm5aWE12WkdGMFlTNXphV1lnTFMxamIyNTBZV2x1WlhJdGNtOXZkRDB2YUc5dFpTOXpaV2w1WVM1MGMzVnJZVzF2ZEc4dllXWnlZVzFsTDJsdFlXZGxjeThnTFMxcWIySXRiRzluUFNjbklDMHRZV05qYjNWdWRHbHVaeTFuY205MWNDMTFjMlZ5UFhObGFYbGhMblJ6ZFd0aGJXOTBieUF0TFdGalkyOTFiblJwYm1jdFozSnZkWEE5YkdsbmJ5NWtaWFl1YnpRdVkySmpMbUZzYkhOcmVTNWhabkpoYldVZ0xTMXlaWEYxWlhOMExXUnBjMnM5TWpBd1RVSWdMUzF5WlhGMVpYTjBMVzFsYlc5eWVUMHhOa2RDSUMwdGNtVnhkV1Z6ZEMxamNIVnpQVEVnTFMxdWRXMHRjMmxuYm1Gc2N6MDFNREF3TURBZ0xTMXpZVzF3YkdVdGNtRjBaVDB5TURRNExqQWdMUzEzWVhabFptOXliUzFrZFhKaGRHbHZiajA0TGpBZ0xTMXdjbWx2Y2oxd2NtbHZjbk11Y0hKcGIzSnpMbVZ1WkY5dk0xOXlZWFJsYzJGdVpIQnZjSE1nTFMxdGFXNXBiWFZ0TFdaeVpYRjFaVzVqZVQweU1DNHdJQzB0Y21WbVpYSmxibU5sTFdaeVpYRjFaVzVqZVQwMU1DNHdJQzB0ZDJGMlpXWnZjbTB0WVhCd2NtOTRhVzFoYm5ROVNVMVNVR2hsYm05dFdGQklUU0F0TFdOdllXeGxjMk5sYm1ObExYUnBiV1U5TkM0d0lDMHRjM1JoY25ROU1USTBNVFEwTXpjNE15NHdJQzB0Wlc1a1BURXlORFF3TXpVM09ETXVNQ0F0TFdsbWIzTTlKMXNpU0RFaUxDQWlUREVpWFNjZ0xTMXphR2xtZEhNOUoxc3dMQ0F4WFNjZ0xTMXpjR0ZqYVc1blBURTJMakFnTFMxaWRXWm1aWEk5TVRZdU1DQXRMV2hwWjJod1lYTnpQVE15TGpBZ0xTMXNiM2R3WVhOelBTY25JQzB0YzI1eUxYUm9jbVZ6YUc5c1pEMDBMakFnTFMxd2MyUXRiR1Z1WjNSb1BUWTBMakFnTFMxelpXVmtQVEV4TWpJZ0xTMXZkWFJ3ZFhRdFpHbHlQUzlvYjIxbEwzTmxhWGxoTG5SemRXdGhiVzkwYnk5aFpuSmhiV1V2Ykc5M0xXaHBaMmd0Y0dGemN5MXRkV3gwYVcxdlpHRnNMMlJoZEdFdmRHVnpkQ0F0TFdOdmJtUnZjaTFrYVhKbFkzUnZjbms5TDJodmJXVXZjMlZwZVdFdWRITjFhMkZ0YjNSdkwyRm1jbUZ0WlM5c2IzY3RhR2xuYUMxd1lYTnpMVzExYkhScGJXOWtZV3d2WTI5dVpHOXlMM1JsYzNScGJtZGZkMkYyWldadmNtMXpJQzB0Ykc5allXd3RjMk5vWldSMWJHVnlQVlJ5ZFdVPSBNVEV6TVE9PSAxIG5vIExRPT1cIlxuICAgICAgICBbJzUnXT1cImFmcmFtZS50YXNrcy5kYXRhLndhdmVmb3Jtcy50ZXN0aW5nIERlcGxveVRlc3RpbmdXYXZlZm9ybXMgTFMxc2IyY3RabWxzWlQxT1QxOVRWRklnTFMxamJHVmhjaTFzYjJkelBVWmhiSE5sSUMwdFpHVjJQVlJ5ZFdVZ0xTMW5jSFZ6UFRBc01pd3pJQzB0YVcxaFoyVTlMMmh2YldVdmMyVnBlV0V1ZEhOMWEyRnRiM1J2TDJGbWNtRnRaUzlwYldGblpYTXZaR0YwWVM1emFXWWdMUzFqYjI1MFlXbHVaWEl0Y205dmREMHZhRzl0WlM5elpXbDVZUzUwYzNWcllXMXZkRzh2WVdaeVlXMWxMMmx0WVdkbGN5OGdMUzFxYjJJdGJHOW5QU2NuSUMwdFlXTmpiM1Z1ZEdsdVp5MW5jbTkxY0MxMWMyVnlQWE5sYVhsaExuUnpkV3RoYlc5MGJ5QXRMV0ZqWTI5MWJuUnBibWN0WjNKdmRYQTliR2xuYnk1a1pYWXVielF1WTJKakxtRnNiSE5yZVM1aFpuSmhiV1VnTFMxeVpYRjFaWE4wTFdScGMyczlNakF3VFVJZ0xTMXlaWEYxWlhOMExXMWxiVzl5ZVQweE5rZENJQzB0Y21WeGRXVnpkQzFqY0hWelBURWdMUzF1ZFcwdGMybG5ibUZzY3owMU1EQXdNREFnTFMxellXMXdiR1V0Y21GMFpUMHlNRFE0TGpBZ0xTMTNZWFpsWm05eWJTMWtkWEpoZEdsdmJqMDRMakFnTFMxd2NtbHZjajF3Y21sdmNuTXVjSEpwYjNKekxtVnVaRjl2TTE5eVlYUmxjMkZ1WkhCdmNITWdMUzF0YVc1cGJYVnRMV1p5WlhGMVpXNWplVDB5TUM0d0lDMHRjbVZtWlhKbGJtTmxMV1p5WlhGMVpXNWplVDAxTUM0d0lDMHRkMkYyWldadmNtMHRZWEJ3Y205NGFXMWhiblE5U1UxU1VHaGxibTl0V0ZCSVRTQXRMV052WVd4bGMyTmxibU5sTFhScGJXVTlOQzR3SUMwdGMzUmhjblE5TVRJME1UUTBNemM0TXk0d0lDMHRaVzVrUFRFeU5EUXdNelUzT0RNdU1DQXRMV2xtYjNNOUoxc2lTREVpTENBaVRERWlYU2NnTFMxemFHbG1kSE05SjFzd0xDQXhYU2NnTFMxemNHRmphVzVuUFRFMkxqQWdMUzFpZFdabVpYSTlNVFl1TUNBdExXaHBaMmh3WVhOelBUTXlMakFnTFMxc2IzZHdZWE56UFNjbklDMHRjMjV5TFhSb2NtVnphRzlzWkQwMExqQWdMUzF3YzJRdGJHVnVaM1JvUFRZMExqQWdMUzF6WldWa1BURXhNaklnTFMxdmRYUndkWFF0WkdseVBTOW9iMjFsTDNObGFYbGhMblJ6ZFd0aGJXOTBieTloWm5KaGJXVXZiRzkzTFdocFoyZ3RjR0Z6Y3kxdGRXeDBhVzF2WkdGc0wyUmhkR0V2ZEdWemRDQXRMV052Ym1SdmNpMWthWEpsWTNSdmNuazlMMmh2YldVdmMyVnBlV0V1ZEhOMWEyRnRiM1J2TDJGbWNtRnRaUzlzYjNjdGFHbG5hQzF3WVhOekxXMTFiSFJwYlc5a1lXd3ZZMjl1Wkc5eUwzUmxjM1JwYm1kZmQyRjJaV1p2Y20xeklDMHRiRzlqWVd3dGMyTm9aV1IxYkdWeVBWUnlkV1U9IE1URXpNZz09IDEgbm8gTFE9PVwiIn0="
    if [ -z "${render_variables}" ]; then
        >&2 echo "empty render variables"
        return "4"
    fi

    # decode
    render_variables="$( echo "${render_variables}" | base64 --decode )"

    # check files to render
    local input_files_render=( law_job_0e5c1d7d9a.sh )
    if [ "${#input_files_render[@]}" == "0" ]; then
        >&2 echo "received empty input files for rendering for LAW_HTCONDOR_JOB_NUMBER ${LAW_HTCONDOR_JOB_NUMBER}"
        return "5"
    fi

    # render files
    local input_file_render
    for input_file_render in ${input_files_render[@]}; do
        # skip if the file refers to _this_ one
        local input_file_render_base="$( basename "${input_file_render}" )"
        [ "${input_file_render_base}" = "${this_file_base}" ] && continue
        # render
        echo "render ${input_file_render}"
        cat > _render.py << EOT
import re
repl = ${render_variables}
repl['input_files_render'] = ''
repl['file_postfix'] = '${file_postfix}' or repl.get('file_postfix', '')
repl['log_file'] = ''
content = open('${input_file_render}', 'r').read()
content = re.sub(r'\{\{(\w+)\}\}', lambda m: repl.get(m.group(1), ''), content)
open('${input_file_render_base}', 'w').write(content)
EOT
        _law_python _render.py
        local render_ret="$?"
        rm -f _render.py
        # handle rendering errors
        if [ "${render_ret}" != "0" ]; then
            >&2 echo "input file rendering failed with code ${render_ret}"
            return "6"
        fi
    done


    #
    # run the actual job file
    #

    # check the job file
    local job_file="law_job_0e5c1d7d9a.sh"
    if [ ! -f "${job_file}" ]; then
        >&2 echo "job file '${job_file}' does not exist"
        return "7"
    fi

    # helper to print a banner
    banner() {
        local msg="$1"

        echo
        echo "================================================================================"
        echo "=== ${msg}"
        echo "================================================================================"
        echo
    }

    # debugging: print its contents
    # echo "=== content of job file '${job_file}'"
    # echo
    # cat "${job_file}"
    # echo
    # echo "=== end of job file content"

    # run it
    banner "Start of law job"

    local job_ret
    bash "${job_file}" ${htcondor_job_arguments}
    job_ret="$?"

    banner "End of law job"

    return "${job_ret}"
}

action() {
    # arguments: file_postfix, log_file
    local file_postfix="$1"
    local log_file="$2"

    # create log directory
    if [ ! -z "${log_file}" ]; then
        local log_dir="$( dirname "${log_file}" )"
        [ ! -d "${log_dir}" ] && mkdir -p "${log_dir}"
    fi

    # run the wrapper function
    if [ -z "${log_file}" ]; then
        htcondor_wrapper "$@"
    elif command -v tee &> /dev/null; then
        set -o pipefail
        echo "---" >> "${log_file}"
        htcondor_wrapper "$@" 2>&1 | tee -a "${log_file}"
    else
        echo "---" >> "${log_file}"
        htcondor_wrapper "$@" &>> "${log_file}"
    fi
}

action "$@"
