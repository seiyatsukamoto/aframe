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
        ['1']="aframe.tasks.infer.infer DeployInferLocal LS1sb2ctZmlsZT1OT19TVFIgLS1jbGVhci1sb2dzPUZhbHNlIC0tZGV2PVRydWUgLS1ncHVzPTAsMSAtLWltYWdlPS9ob21lL3NlaXlhLnRzdWthbW90by9hZnJhbWUvaW1hZ2VzL2luZmVyLnNpZiAtLWNvbnRhaW5lci1yb290PS9ob21lL3NlaXlhLnRzdWthbW90by9hZnJhbWUvaW1hZ2VzLyAtLWFjY291bnRpbmctZ3JvdXAtdXNlcj1zZWl5YS50c3VrYW1vdG8gLS1hY2NvdW50aW5nLWdyb3VwPWxpZ28uZGV2Lm80LmNiYy5hbGxza3kuYWZyYW1lIC0tcmVxdWVzdC1kaXNrPTFHIC0tcmVxdWVzdC1tZW1vcnk9NkcgLS1yZXF1ZXN0LWNwdXM9MSAtLWlmb3M9J1siSDEiLCAiTDEiXScgLS1pbmZlcmVuY2Utc2FtcGxpbmctcmF0ZT00LjAgLS1iYXRjaC1zaXplPTEyOCAtLXBzZC1sZW5ndGg9NjQuMCAtLWNsdXN0ZXItd2luZG93LWxlbmd0aD04LjAgLS1pbnRlZ3JhdGlvbi13aW5kb3ctbGVuZ3RoPTEuNSAtLWZkdXJhdGlvbj0xLjAgLS1UYj0zMTUzNjAwMC4wIC0tc2hpZnRzPSdbMCwgMV0nIC0tc2VxdWVuY2UtaWQ9MTAwMSAtLW1vZGVsLW5hbWU9YWZyYW1lLXN0cmVhbSAtLW1vZGVsLXZlcnNpb249LTEgLS1zdHJlYW1zLXBlci1ncHU9NiAtLXJhdGUtcGVyLWdwdT03MC4wIC0temVyby1sYWc9VHJ1ZSAtLXJldHVybi10aW1lc2VyaWVzPUZhbHNlIC0tb3V0cHV0LWRpcj0vaG9tZS9zZWl5YS50c3VrYW1vdG8vYWZyYW1lL2xheWVyZWQvYmFuZF8xL3Jlc3VsdHMgLS10cmFpbi10YXNrPVRyYWluIC0tY29uZG9yLWRpcmVjdG9yeT0vaG9tZS9zZWl5YS50c3VrYW1vdG8vYWZyYW1lL2xheWVyZWQvY29uZG9yL2luZmVyIC0tdHJpdG9uLWltYWdlPWhlcm1lcy90cml0b25zZXJ2ZXI6MjMuMDEgLS1sb2NhbC1zY2hlZHVsZXI9VHJ1ZQ== MjA1OQ== 1 no LQ=="
        ['2']="aframe.tasks.infer.infer DeployInferLocal LS1sb2ctZmlsZT1OT19TVFIgLS1jbGVhci1sb2dzPUZhbHNlIC0tZGV2PVRydWUgLS1ncHVzPTAsMSAtLWltYWdlPS9ob21lL3NlaXlhLnRzdWthbW90by9hZnJhbWUvaW1hZ2VzL2luZmVyLnNpZiAtLWNvbnRhaW5lci1yb290PS9ob21lL3NlaXlhLnRzdWthbW90by9hZnJhbWUvaW1hZ2VzLyAtLWFjY291bnRpbmctZ3JvdXAtdXNlcj1zZWl5YS50c3VrYW1vdG8gLS1hY2NvdW50aW5nLWdyb3VwPWxpZ28uZGV2Lm80LmNiYy5hbGxza3kuYWZyYW1lIC0tcmVxdWVzdC1kaXNrPTFHIC0tcmVxdWVzdC1tZW1vcnk9NkcgLS1yZXF1ZXN0LWNwdXM9MSAtLWlmb3M9J1siSDEiLCAiTDEiXScgLS1pbmZlcmVuY2Utc2FtcGxpbmctcmF0ZT00LjAgLS1iYXRjaC1zaXplPTEyOCAtLXBzZC1sZW5ndGg9NjQuMCAtLWNsdXN0ZXItd2luZG93LWxlbmd0aD04LjAgLS1pbnRlZ3JhdGlvbi13aW5kb3ctbGVuZ3RoPTEuNSAtLWZkdXJhdGlvbj0xLjAgLS1UYj0zMTUzNjAwMC4wIC0tc2hpZnRzPSdbMCwgMV0nIC0tc2VxdWVuY2UtaWQ9MTAwMSAtLW1vZGVsLW5hbWU9YWZyYW1lLXN0cmVhbSAtLW1vZGVsLXZlcnNpb249LTEgLS1zdHJlYW1zLXBlci1ncHU9NiAtLXJhdGUtcGVyLWdwdT03MC4wIC0temVyby1sYWc9VHJ1ZSAtLXJldHVybi10aW1lc2VyaWVzPUZhbHNlIC0tb3V0cHV0LWRpcj0vaG9tZS9zZWl5YS50c3VrYW1vdG8vYWZyYW1lL2xheWVyZWQvYmFuZF8xL3Jlc3VsdHMgLS10cmFpbi10YXNrPVRyYWluIC0tY29uZG9yLWRpcmVjdG9yeT0vaG9tZS9zZWl5YS50c3VrYW1vdG8vYWZyYW1lL2xheWVyZWQvY29uZG9yL2luZmVyIC0tdHJpdG9uLWltYWdlPWhlcm1lcy90cml0b25zZXJ2ZXI6MjMuMDEgLS1sb2NhbC1zY2hlZHVsZXI9VHJ1ZQ== MjA2MA== 1 no LQ=="
        ['3']="aframe.tasks.infer.infer DeployInferLocal LS1sb2ctZmlsZT1OT19TVFIgLS1jbGVhci1sb2dzPUZhbHNlIC0tZGV2PVRydWUgLS1ncHVzPTAsMSAtLWltYWdlPS9ob21lL3NlaXlhLnRzdWthbW90by9hZnJhbWUvaW1hZ2VzL2luZmVyLnNpZiAtLWNvbnRhaW5lci1yb290PS9ob21lL3NlaXlhLnRzdWthbW90by9hZnJhbWUvaW1hZ2VzLyAtLWFjY291bnRpbmctZ3JvdXAtdXNlcj1zZWl5YS50c3VrYW1vdG8gLS1hY2NvdW50aW5nLWdyb3VwPWxpZ28uZGV2Lm80LmNiYy5hbGxza3kuYWZyYW1lIC0tcmVxdWVzdC1kaXNrPTFHIC0tcmVxdWVzdC1tZW1vcnk9NkcgLS1yZXF1ZXN0LWNwdXM9MSAtLWlmb3M9J1siSDEiLCAiTDEiXScgLS1pbmZlcmVuY2Utc2FtcGxpbmctcmF0ZT00LjAgLS1iYXRjaC1zaXplPTEyOCAtLXBzZC1sZW5ndGg9NjQuMCAtLWNsdXN0ZXItd2luZG93LWxlbmd0aD04LjAgLS1pbnRlZ3JhdGlvbi13aW5kb3ctbGVuZ3RoPTEuNSAtLWZkdXJhdGlvbj0xLjAgLS1UYj0zMTUzNjAwMC4wIC0tc2hpZnRzPSdbMCwgMV0nIC0tc2VxdWVuY2UtaWQ9MTAwMSAtLW1vZGVsLW5hbWU9YWZyYW1lLXN0cmVhbSAtLW1vZGVsLXZlcnNpb249LTEgLS1zdHJlYW1zLXBlci1ncHU9NiAtLXJhdGUtcGVyLWdwdT03MC4wIC0temVyby1sYWc9VHJ1ZSAtLXJldHVybi10aW1lc2VyaWVzPUZhbHNlIC0tb3V0cHV0LWRpcj0vaG9tZS9zZWl5YS50c3VrYW1vdG8vYWZyYW1lL2xheWVyZWQvYmFuZF8xL3Jlc3VsdHMgLS10cmFpbi10YXNrPVRyYWluIC0tY29uZG9yLWRpcmVjdG9yeT0vaG9tZS9zZWl5YS50c3VrYW1vdG8vYWZyYW1lL2xheWVyZWQvY29uZG9yL2luZmVyIC0tdHJpdG9uLWltYWdlPWhlcm1lcy90cml0b25zZXJ2ZXI6MjMuMDEgLS1sb2NhbC1zY2hlZHVsZXI9VHJ1ZQ== MjA2MQ== 1 no LQ=="
        ['4']="aframe.tasks.infer.infer DeployInferLocal LS1sb2ctZmlsZT1OT19TVFIgLS1jbGVhci1sb2dzPUZhbHNlIC0tZGV2PVRydWUgLS1ncHVzPTAsMSAtLWltYWdlPS9ob21lL3NlaXlhLnRzdWthbW90by9hZnJhbWUvaW1hZ2VzL2luZmVyLnNpZiAtLWNvbnRhaW5lci1yb290PS9ob21lL3NlaXlhLnRzdWthbW90by9hZnJhbWUvaW1hZ2VzLyAtLWFjY291bnRpbmctZ3JvdXAtdXNlcj1zZWl5YS50c3VrYW1vdG8gLS1hY2NvdW50aW5nLWdyb3VwPWxpZ28uZGV2Lm80LmNiYy5hbGxza3kuYWZyYW1lIC0tcmVxdWVzdC1kaXNrPTFHIC0tcmVxdWVzdC1tZW1vcnk9NkcgLS1yZXF1ZXN0LWNwdXM9MSAtLWlmb3M9J1siSDEiLCAiTDEiXScgLS1pbmZlcmVuY2Utc2FtcGxpbmctcmF0ZT00LjAgLS1iYXRjaC1zaXplPTEyOCAtLXBzZC1sZW5ndGg9NjQuMCAtLWNsdXN0ZXItd2luZG93LWxlbmd0aD04LjAgLS1pbnRlZ3JhdGlvbi13aW5kb3ctbGVuZ3RoPTEuNSAtLWZkdXJhdGlvbj0xLjAgLS1UYj0zMTUzNjAwMC4wIC0tc2hpZnRzPSdbMCwgMV0nIC0tc2VxdWVuY2UtaWQ9MTAwMSAtLW1vZGVsLW5hbWU9YWZyYW1lLXN0cmVhbSAtLW1vZGVsLXZlcnNpb249LTEgLS1zdHJlYW1zLXBlci1ncHU9NiAtLXJhdGUtcGVyLWdwdT03MC4wIC0temVyby1sYWc9VHJ1ZSAtLXJldHVybi10aW1lc2VyaWVzPUZhbHNlIC0tb3V0cHV0LWRpcj0vaG9tZS9zZWl5YS50c3VrYW1vdG8vYWZyYW1lL2xheWVyZWQvYmFuZF8xL3Jlc3VsdHMgLS10cmFpbi10YXNrPVRyYWluIC0tY29uZG9yLWRpcmVjdG9yeT0vaG9tZS9zZWl5YS50c3VrYW1vdG8vYWZyYW1lL2xheWVyZWQvY29uZG9yL2luZmVyIC0tdHJpdG9uLWltYWdlPWhlcm1lcy90cml0b25zZXJ2ZXI6MjMuMDEgLS1sb2NhbC1zY2hlZHVsZXI9VHJ1ZQ== MjA2Mg== 1 no LQ=="
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
    local render_variables="eyJqb2JfZmlsZSI6ICJsYXdfam9iXzhhNjk1YmFiZmIuc2giLCAiZXhlY3V0YWJsZV9maWxlIjogImh0Y29uZG9yX3dyYXBwZXJfMjI1NjNkMjJjM185MjAuc2giLCAiaW5wdXRfZmlsZXMiOiAibGF3X2pvYl84YTY5NWJhYmZiLnNoIGh0Y29uZG9yX3dyYXBwZXJfMjI1NjNkMjJjM185MjAuc2giLCAiaW5wdXRfZmlsZXNfcmVuZGVyIjogImxhd19qb2JfOGE2OTViYWJmYi5zaCIsICJodGNvbmRvcl9qb2JfYXJndW1lbnRzX21hcCI6ICJbJzEnXT1cImFmcmFtZS50YXNrcy5pbmZlci5pbmZlciBEZXBsb3lJbmZlckxvY2FsIExTMXNiMmN0Wm1sc1pUMU9UMTlUVkZJZ0xTMWpiR1ZoY2kxc2IyZHpQVVpoYkhObElDMHRaR1YyUFZSeWRXVWdMUzFuY0hWelBUQXNNU0F0TFdsdFlXZGxQUzlvYjIxbEwzTmxhWGxoTG5SemRXdGhiVzkwYnk5aFpuSmhiV1V2YVcxaFoyVnpMMmx1Wm1WeUxuTnBaaUF0TFdOdmJuUmhhVzVsY2kxeWIyOTBQUzlvYjIxbEwzTmxhWGxoTG5SemRXdGhiVzkwYnk5aFpuSmhiV1V2YVcxaFoyVnpMeUF0TFdGalkyOTFiblJwYm1jdFozSnZkWEF0ZFhObGNqMXpaV2w1WVM1MGMzVnJZVzF2ZEc4Z0xTMWhZMk52ZFc1MGFXNW5MV2R5YjNWd1BXeHBaMjh1WkdWMkxtODBMbU5pWXk1aGJHeHphM2t1WVdaeVlXMWxJQzB0Y21WeGRXVnpkQzFrYVhOclBURkhJQzB0Y21WeGRXVnpkQzF0WlcxdmNuazlOa2NnTFMxeVpYRjFaWE4wTFdOd2RYTTlNU0F0TFdsbWIzTTlKMXNpU0RFaUxDQWlUREVpWFNjZ0xTMXBibVpsY21WdVkyVXRjMkZ0Y0d4cGJtY3RjbUYwWlQwMExqQWdMUzFpWVhSamFDMXphWHBsUFRFeU9DQXRMWEJ6WkMxc1pXNW5kR2c5TmpRdU1DQXRMV05zZFhOMFpYSXRkMmx1Wkc5M0xXeGxibWQwYUQwNExqQWdMUzFwYm5SbFozSmhkR2x2YmkxM2FXNWtiM2N0YkdWdVozUm9QVEV1TlNBdExXWmtkWEpoZEdsdmJqMHhMakFnTFMxVVlqMHpNVFV6TmpBd01DNHdJQzB0YzJocFpuUnpQU2RiTUN3Z01WMG5JQzB0YzJWeGRXVnVZMlV0YVdROU1UQXdNU0F0TFcxdlpHVnNMVzVoYldVOVlXWnlZVzFsTFhOMGNtVmhiU0F0TFcxdlpHVnNMWFpsY25OcGIyNDlMVEVnTFMxemRISmxZVzF6TFhCbGNpMW5jSFU5TmlBdExYSmhkR1V0Y0dWeUxXZHdkVDAzTUM0d0lDMHRlbVZ5Ynkxc1lXYzlWSEoxWlNBdExYSmxkSFZ5YmkxMGFXMWxjMlZ5YVdWelBVWmhiSE5sSUMwdGIzVjBjSFYwTFdScGNqMHZhRzl0WlM5elpXbDVZUzUwYzNWcllXMXZkRzh2WVdaeVlXMWxMMnhoZVdWeVpXUXZZbUZ1WkY4eEwzSmxjM1ZzZEhNZ0xTMTBjbUZwYmkxMFlYTnJQVlJ5WVdsdUlDMHRZMjl1Wkc5eUxXUnBjbVZqZEc5eWVUMHZhRzl0WlM5elpXbDVZUzUwYzNWcllXMXZkRzh2WVdaeVlXMWxMMnhoZVdWeVpXUXZZMjl1Wkc5eUwybHVabVZ5SUMwdGRISnBkRzl1TFdsdFlXZGxQV2hsY20xbGN5OTBjbWwwYjI1elpYSjJaWEk2TWpNdU1ERWdMUzFzYjJOaGJDMXpZMmhsWkhWc1pYSTlWSEoxWlE9PSBNakExT1E9PSAxIG5vIExRPT1cIlxuICAgICAgICBbJzInXT1cImFmcmFtZS50YXNrcy5pbmZlci5pbmZlciBEZXBsb3lJbmZlckxvY2FsIExTMXNiMmN0Wm1sc1pUMU9UMTlUVkZJZ0xTMWpiR1ZoY2kxc2IyZHpQVVpoYkhObElDMHRaR1YyUFZSeWRXVWdMUzFuY0hWelBUQXNNU0F0TFdsdFlXZGxQUzlvYjIxbEwzTmxhWGxoTG5SemRXdGhiVzkwYnk5aFpuSmhiV1V2YVcxaFoyVnpMMmx1Wm1WeUxuTnBaaUF0TFdOdmJuUmhhVzVsY2kxeWIyOTBQUzlvYjIxbEwzTmxhWGxoTG5SemRXdGhiVzkwYnk5aFpuSmhiV1V2YVcxaFoyVnpMeUF0TFdGalkyOTFiblJwYm1jdFozSnZkWEF0ZFhObGNqMXpaV2w1WVM1MGMzVnJZVzF2ZEc4Z0xTMWhZMk52ZFc1MGFXNW5MV2R5YjNWd1BXeHBaMjh1WkdWMkxtODBMbU5pWXk1aGJHeHphM2t1WVdaeVlXMWxJQzB0Y21WeGRXVnpkQzFrYVhOclBURkhJQzB0Y21WeGRXVnpkQzF0WlcxdmNuazlOa2NnTFMxeVpYRjFaWE4wTFdOd2RYTTlNU0F0TFdsbWIzTTlKMXNpU0RFaUxDQWlUREVpWFNjZ0xTMXBibVpsY21WdVkyVXRjMkZ0Y0d4cGJtY3RjbUYwWlQwMExqQWdMUzFpWVhSamFDMXphWHBsUFRFeU9DQXRMWEJ6WkMxc1pXNW5kR2c5TmpRdU1DQXRMV05zZFhOMFpYSXRkMmx1Wkc5M0xXeGxibWQwYUQwNExqQWdMUzFwYm5SbFozSmhkR2x2YmkxM2FXNWtiM2N0YkdWdVozUm9QVEV1TlNBdExXWmtkWEpoZEdsdmJqMHhMakFnTFMxVVlqMHpNVFV6TmpBd01DNHdJQzB0YzJocFpuUnpQU2RiTUN3Z01WMG5JQzB0YzJWeGRXVnVZMlV0YVdROU1UQXdNU0F0TFcxdlpHVnNMVzVoYldVOVlXWnlZVzFsTFhOMGNtVmhiU0F0TFcxdlpHVnNMWFpsY25OcGIyNDlMVEVnTFMxemRISmxZVzF6TFhCbGNpMW5jSFU5TmlBdExYSmhkR1V0Y0dWeUxXZHdkVDAzTUM0d0lDMHRlbVZ5Ynkxc1lXYzlWSEoxWlNBdExYSmxkSFZ5YmkxMGFXMWxjMlZ5YVdWelBVWmhiSE5sSUMwdGIzVjBjSFYwTFdScGNqMHZhRzl0WlM5elpXbDVZUzUwYzNWcllXMXZkRzh2WVdaeVlXMWxMMnhoZVdWeVpXUXZZbUZ1WkY4eEwzSmxjM1ZzZEhNZ0xTMTBjbUZwYmkxMFlYTnJQVlJ5WVdsdUlDMHRZMjl1Wkc5eUxXUnBjbVZqZEc5eWVUMHZhRzl0WlM5elpXbDVZUzUwYzNWcllXMXZkRzh2WVdaeVlXMWxMMnhoZVdWeVpXUXZZMjl1Wkc5eUwybHVabVZ5SUMwdGRISnBkRzl1TFdsdFlXZGxQV2hsY20xbGN5OTBjbWwwYjI1elpYSjJaWEk2TWpNdU1ERWdMUzFzYjJOaGJDMXpZMmhsWkhWc1pYSTlWSEoxWlE9PSBNakEyTUE9PSAxIG5vIExRPT1cIlxuICAgICAgICBbJzMnXT1cImFmcmFtZS50YXNrcy5pbmZlci5pbmZlciBEZXBsb3lJbmZlckxvY2FsIExTMXNiMmN0Wm1sc1pUMU9UMTlUVkZJZ0xTMWpiR1ZoY2kxc2IyZHpQVVpoYkhObElDMHRaR1YyUFZSeWRXVWdMUzFuY0hWelBUQXNNU0F0TFdsdFlXZGxQUzlvYjIxbEwzTmxhWGxoTG5SemRXdGhiVzkwYnk5aFpuSmhiV1V2YVcxaFoyVnpMMmx1Wm1WeUxuTnBaaUF0TFdOdmJuUmhhVzVsY2kxeWIyOTBQUzlvYjIxbEwzTmxhWGxoTG5SemRXdGhiVzkwYnk5aFpuSmhiV1V2YVcxaFoyVnpMeUF0TFdGalkyOTFiblJwYm1jdFozSnZkWEF0ZFhObGNqMXpaV2w1WVM1MGMzVnJZVzF2ZEc4Z0xTMWhZMk52ZFc1MGFXNW5MV2R5YjNWd1BXeHBaMjh1WkdWMkxtODBMbU5pWXk1aGJHeHphM2t1WVdaeVlXMWxJQzB0Y21WeGRXVnpkQzFrYVhOclBURkhJQzB0Y21WeGRXVnpkQzF0WlcxdmNuazlOa2NnTFMxeVpYRjFaWE4wTFdOd2RYTTlNU0F0TFdsbWIzTTlKMXNpU0RFaUxDQWlUREVpWFNjZ0xTMXBibVpsY21WdVkyVXRjMkZ0Y0d4cGJtY3RjbUYwWlQwMExqQWdMUzFpWVhSamFDMXphWHBsUFRFeU9DQXRMWEJ6WkMxc1pXNW5kR2c5TmpRdU1DQXRMV05zZFhOMFpYSXRkMmx1Wkc5M0xXeGxibWQwYUQwNExqQWdMUzFwYm5SbFozSmhkR2x2YmkxM2FXNWtiM2N0YkdWdVozUm9QVEV1TlNBdExXWmtkWEpoZEdsdmJqMHhMakFnTFMxVVlqMHpNVFV6TmpBd01DNHdJQzB0YzJocFpuUnpQU2RiTUN3Z01WMG5JQzB0YzJWeGRXVnVZMlV0YVdROU1UQXdNU0F0TFcxdlpHVnNMVzVoYldVOVlXWnlZVzFsTFhOMGNtVmhiU0F0TFcxdlpHVnNMWFpsY25OcGIyNDlMVEVnTFMxemRISmxZVzF6TFhCbGNpMW5jSFU5TmlBdExYSmhkR1V0Y0dWeUxXZHdkVDAzTUM0d0lDMHRlbVZ5Ynkxc1lXYzlWSEoxWlNBdExYSmxkSFZ5YmkxMGFXMWxjMlZ5YVdWelBVWmhiSE5sSUMwdGIzVjBjSFYwTFdScGNqMHZhRzl0WlM5elpXbDVZUzUwYzNWcllXMXZkRzh2WVdaeVlXMWxMMnhoZVdWeVpXUXZZbUZ1WkY4eEwzSmxjM1ZzZEhNZ0xTMTBjbUZwYmkxMFlYTnJQVlJ5WVdsdUlDMHRZMjl1Wkc5eUxXUnBjbVZqZEc5eWVUMHZhRzl0WlM5elpXbDVZUzUwYzNWcllXMXZkRzh2WVdaeVlXMWxMMnhoZVdWeVpXUXZZMjl1Wkc5eUwybHVabVZ5SUMwdGRISnBkRzl1TFdsdFlXZGxQV2hsY20xbGN5OTBjbWwwYjI1elpYSjJaWEk2TWpNdU1ERWdMUzFzYjJOaGJDMXpZMmhsWkhWc1pYSTlWSEoxWlE9PSBNakEyTVE9PSAxIG5vIExRPT1cIlxuICAgICAgICBbJzQnXT1cImFmcmFtZS50YXNrcy5pbmZlci5pbmZlciBEZXBsb3lJbmZlckxvY2FsIExTMXNiMmN0Wm1sc1pUMU9UMTlUVkZJZ0xTMWpiR1ZoY2kxc2IyZHpQVVpoYkhObElDMHRaR1YyUFZSeWRXVWdMUzFuY0hWelBUQXNNU0F0TFdsdFlXZGxQUzlvYjIxbEwzTmxhWGxoTG5SemRXdGhiVzkwYnk5aFpuSmhiV1V2YVcxaFoyVnpMMmx1Wm1WeUxuTnBaaUF0TFdOdmJuUmhhVzVsY2kxeWIyOTBQUzlvYjIxbEwzTmxhWGxoTG5SemRXdGhiVzkwYnk5aFpuSmhiV1V2YVcxaFoyVnpMeUF0TFdGalkyOTFiblJwYm1jdFozSnZkWEF0ZFhObGNqMXpaV2w1WVM1MGMzVnJZVzF2ZEc4Z0xTMWhZMk52ZFc1MGFXNW5MV2R5YjNWd1BXeHBaMjh1WkdWMkxtODBMbU5pWXk1aGJHeHphM2t1WVdaeVlXMWxJQzB0Y21WeGRXVnpkQzFrYVhOclBURkhJQzB0Y21WeGRXVnpkQzF0WlcxdmNuazlOa2NnTFMxeVpYRjFaWE4wTFdOd2RYTTlNU0F0TFdsbWIzTTlKMXNpU0RFaUxDQWlUREVpWFNjZ0xTMXBibVpsY21WdVkyVXRjMkZ0Y0d4cGJtY3RjbUYwWlQwMExqQWdMUzFpWVhSamFDMXphWHBsUFRFeU9DQXRMWEJ6WkMxc1pXNW5kR2c5TmpRdU1DQXRMV05zZFhOMFpYSXRkMmx1Wkc5M0xXeGxibWQwYUQwNExqQWdMUzFwYm5SbFozSmhkR2x2YmkxM2FXNWtiM2N0YkdWdVozUm9QVEV1TlNBdExXWmtkWEpoZEdsdmJqMHhMakFnTFMxVVlqMHpNVFV6TmpBd01DNHdJQzB0YzJocFpuUnpQU2RiTUN3Z01WMG5JQzB0YzJWeGRXVnVZMlV0YVdROU1UQXdNU0F0TFcxdlpHVnNMVzVoYldVOVlXWnlZVzFsTFhOMGNtVmhiU0F0TFcxdlpHVnNMWFpsY25OcGIyNDlMVEVnTFMxemRISmxZVzF6TFhCbGNpMW5jSFU5TmlBdExYSmhkR1V0Y0dWeUxXZHdkVDAzTUM0d0lDMHRlbVZ5Ynkxc1lXYzlWSEoxWlNBdExYSmxkSFZ5YmkxMGFXMWxjMlZ5YVdWelBVWmhiSE5sSUMwdGIzVjBjSFYwTFdScGNqMHZhRzl0WlM5elpXbDVZUzUwYzNWcllXMXZkRzh2WVdaeVlXMWxMMnhoZVdWeVpXUXZZbUZ1WkY4eEwzSmxjM1ZzZEhNZ0xTMTBjbUZwYmkxMFlYTnJQVlJ5WVdsdUlDMHRZMjl1Wkc5eUxXUnBjbVZqZEc5eWVUMHZhRzl0WlM5elpXbDVZUzUwYzNWcllXMXZkRzh2WVdaeVlXMWxMMnhoZVdWeVpXUXZZMjl1Wkc5eUwybHVabVZ5SUMwdGRISnBkRzl1TFdsdFlXZGxQV2hsY20xbGN5OTBjbWwwYjI1elpYSjJaWEk2TWpNdU1ERWdMUzFzYjJOaGJDMXpZMmhsWkhWc1pYSTlWSEoxWlE9PSBNakEyTWc9PSAxIG5vIExRPT1cIiJ9"
    if [ -z "${render_variables}" ]; then
        >&2 echo "empty render variables"
        return "4"
    fi

    # decode
    render_variables="$( echo "${render_variables}" | base64 --decode )"

    # check files to render
    local input_files_render=( law_job_8a695babfb.sh )
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
    local job_file="law_job_8a695babfb.sh"
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
