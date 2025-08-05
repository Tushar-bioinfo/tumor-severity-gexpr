#!/usr/bin/env nextflow

params.split_variants = ['5pct_80', '10pct_80', '25pct_80']
params.batch_sizes    = [16,32,64]
params.model_types    = ['1layer', '3layer']
params.script         = 'cnn_optuna_eval.py'

process run_optuna_eval {
    tag "${split}_${batch_size}_${model_type}"
    publishDir "results/${split}/batch${batch_size}/${model_type}", mode: 'copy'

    input:
    val split
    val batch_size
    val model_type
    path params.script

    output:
    file "*"

    script:
    """
    python ${params.script} \
        --split ${split} \
        --batch_size ${batch_size} \
        --model_type ${model_type}
    """
}

workflow {
    Channel
        .from(params.split_variants)
        .combine(params.batch_sizes)
        .combine(params.model_types)
        .map { (split_bs, model_type) -> 
            def (split, batch_size) = split_bs
            tuple(split, batch_size, model_type)
        }
        | run_optuna_eval
}
