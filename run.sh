# start gcp and recommender app
REC=/opt/recommender

$REC/gcp/cloud_sql_proxy -instances=fiml-1:europe-west2:fiml=tcp:5432 -credential_file=$REC/gcp/fiml-1-8fb931cba067.json > /opt/bla.log 2>&1 & 

python -m reclibwh.apps.serve_recommender --postgres_config $REC/gcp/gcp.postgres.config --model_ex $REC/deploy/2020_06_04_MF_latest_trained_reduced/  --model_im $REC/deploy/2020_06_04_ALS_latest_trained_reduced/ --port 80 > /opt/bla.log 2>&1 &

tail -f  /opt/bla.log