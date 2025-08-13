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
        ['1']="aframe.tasks.infer.infer DeployInferLocal LS1sb2ctZmlsZT1OT19TVFIgLS1jbGVhci1sb2dzPUZhbHNlIC0tZGV2PVRydWUgLS1ncHVzPTAsMSAtLWltYWdlPS9ob21lL3NlaXlhLnRzdWthbW90by9hZnJhbWUvaW1hZ2VzL2luZmVyLnNpZiAtLWNvbnRhaW5lci1yb290PS9ob21lL3NlaXlhLnRzdWthbW90by9hZnJhbWUvaW1hZ2VzLyAtLWFjY291bnRpbmctZ3JvdXAtdXNlcj1zZWl5YS50c3VrYW1vdG8gLS1hY2NvdW50aW5nLWdyb3VwPWxpZ28uZGV2Lm80LmNiYy5hbGxza3kuYWZyYW1lIC0tcmVxdWVzdC1kaXNrPTFHIC0tcmVxdWVzdC1tZW1vcnk9NkcgLS1yZXF1ZXN0LWNwdXM9MSAtLWlmb3M9J1siSDEiLCAiTDEiXScgLS1pbmZlcmVuY2Utc2FtcGxpbmctcmF0ZT00LjAgLS1iYXRjaC1zaXplPTEyOCAtLXBzZC1sZW5ndGg9NjQuMCAtLWNsdXN0ZXItd2luZG93LWxlbmd0aD04LjAgLS1pbnRlZ3JhdGlvbi13aW5kb3ctbGVuZ3RoPTEuNSAtLWZkdXJhdGlvbj0xLjAgLS1UYj0zMTUzNjAwMC4wIC0tc2hpZnRzPSdbMCwgMV0nIC0tc2VxdWVuY2UtaWQ9MTAwMSAtLW1vZGVsLW5hbWU9YWZyYW1lLXN0cmVhbSAtLW1vZGVsLXZlcnNpb249LTEgLS1zdHJlYW1zLXBlci1ncHU9NiAtLXJhdGUtcGVyLWdwdT03MC4wIC0temVyby1sYWc9VHJ1ZSAtLXJldHVybi10aW1lc2VyaWVzPUZhbHNlIC0tb3V0cHV0LWRpcj0vaG9tZS9zZWl5YS50c3VrYW1vdG8vYWZyYW1lL2xheWVyZWQvYmFuZF8yL3Jlc3VsdHMgLS10cmFpbi10YXNrPVRyYWluIC0tY29uZG9yLWRpcmVjdG9yeT0vaG9tZS9zZWl5YS50c3VrYW1vdG8vYWZyYW1lL2xheWVyZWQvY29uZG9yL2luZmVyIC0tdHJpdG9uLWltYWdlPWhlcm1lcy90cml0b25zZXJ2ZXI6MjMuMDEgLS1sb2NhbC1zY2hlZHVsZXI9VHJ1ZQ== MTg3Mg== 1 no LQ=="
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
    local render_variables="eyJqb2JfZmlsZSI6ICJsYXdfam9iXzhhNjk1YmFiZmIuc2giLCAiZXhlY3V0YWJsZV9maWxlIjogImh0Y29uZG9yX3dyYXBwZXJfMjI1NjNkMjJjM180MjUyLnNoIiwgImlucHV0X2ZpbGVzIjogImxhd19qb2JfOGE2OTViYWJmYi5zaCBodGNvbmRvcl93cmFwcGVyXzIyNTYzZDIyYzNfNDI1Mi5zaCIsICJpbnB1dF9maWxlc19yZW5kZXIiOiAibGF3X2pvYl84YTY5NWJhYmZiLnNoIiwgImh0Y29uZG9yX2pvYl9hcmd1bWVudHNfbWFwIjogIlsnMSddPVwiYWZyYW1lLnRhc2tzLmluZmVyLmluZmVyIERlcGxveUluZmVyTG9jYWwgTFMxc2IyY3RabWxzWlQxT1QxOVRWRklnTFMxamJHVmhjaTFzYjJkelBVWmhiSE5sSUMwdFpHVjJQVlJ5ZFdVZ0xTMW5jSFZ6UFRBc01TQXRMV2x0WVdkbFBTOW9iMjFsTDNObGFYbGhMblJ6ZFd0aGJXOTBieTloWm5KaGJXVXZhVzFoWjJWekwybHVabVZ5TG5OcFppQXRMV052Ym5SaGFXNWxjaTF5YjI5MFBTOW9iMjFsTDNObGFYbGhMblJ6ZFd0aGJXOTBieTloWm5KaGJXVXZhVzFoWjJWekx5QXRMV0ZqWTI5MWJuUnBibWN0WjNKdmRYQXRkWE5sY2oxelpXbDVZUzUwYzNWcllXMXZkRzhnTFMxaFkyTnZkVzUwYVc1bkxXZHliM1Z3UFd4cFoyOHVaR1YyTG04MExtTmlZeTVoYkd4emEza3VZV1p5WVcxbElDMHRjbVZ4ZFdWemRDMWthWE5yUFRGSElDMHRjbVZ4ZFdWemRDMXRaVzF2Y25rOU5rY2dMUzF5WlhGMVpYTjBMV053ZFhNOU1TQXRMV2xtYjNNOUoxc2lTREVpTENBaVRERWlYU2NnTFMxcGJtWmxjbVZ1WTJVdGMyRnRjR3hwYm1jdGNtRjBaVDAwTGpBZ0xTMWlZWFJqYUMxemFYcGxQVEV5T0NBdExYQnpaQzFzWlc1bmRHZzlOalF1TUNBdExXTnNkWE4wWlhJdGQybHVaRzkzTFd4bGJtZDBhRDA0TGpBZ0xTMXBiblJsWjNKaGRHbHZiaTEzYVc1a2IzY3RiR1Z1WjNSb1BURXVOU0F0TFdaa2RYSmhkR2x2YmoweExqQWdMUzFVWWowek1UVXpOakF3TUM0d0lDMHRjMmhwWm5SelBTZGJNQ3dnTVYwbklDMHRjMlZ4ZFdWdVkyVXRhV1E5TVRBd01TQXRMVzF2WkdWc0xXNWhiV1U5WVdaeVlXMWxMWE4wY21WaGJTQXRMVzF2WkdWc0xYWmxjbk5wYjI0OUxURWdMUzF6ZEhKbFlXMXpMWEJsY2kxbmNIVTlOaUF0TFhKaGRHVXRjR1Z5TFdkd2RUMDNNQzR3SUMwdGVtVnlieTFzWVdjOVZISjFaU0F0TFhKbGRIVnliaTEwYVcxbGMyVnlhV1Z6UFVaaGJITmxJQzB0YjNWMGNIVjBMV1JwY2owdmFHOXRaUzl6WldsNVlTNTBjM1ZyWVcxdmRHOHZZV1p5WVcxbEwyeGhlV1Z5WldRdlltRnVaRjh5TDNKbGMzVnNkSE1nTFMxMGNtRnBiaTEwWVhOclBWUnlZV2x1SUMwdFkyOXVaRzl5TFdScGNtVmpkRzl5ZVQwdmFHOXRaUzl6WldsNVlTNTBjM1ZyWVcxdmRHOHZZV1p5WVcxbEwyeGhlV1Z5WldRdlkyOXVaRzl5TDJsdVptVnlJQzB0ZEhKcGRHOXVMV2x0WVdkbFBXaGxjbTFsY3k5MGNtbDBiMjV6WlhKMlpYSTZNak11TURFZ0xTMXNiMk5oYkMxelkyaGxaSFZzWlhJOVZISjFaUT09IE1UZzNNZz09IDEgbm8gTFE9PVwiIn0="
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
