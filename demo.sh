valohai-local-run load --output-path "../data" && \
valohai-local-run transform --data-file ../data/demo_data_train.json --output-path ../data/cleansed_data.json --adhoc && \
valohai-local-run cooccurrence --cleansed-cvs "../data/cleansed_data.json" --output-path "../data" && \
python3 src/main.py train_glove --counter "../data/counter.pkl" --cooccurrence-matrix "../data/cooccurrence_matrix.npz" --output-path "../data/glove_embedder.pkl" && \
python3 src/main.py apply --glove-embedder "../data/glove_embedder.pkl" --labeled-docs-path "../data/demo_data_test.json" --output-path "../data/embedded_data.json" && \
python3 src/main.py evaluate --embedded-cvs-labeled "../data/embedded_data.json" --labeled-docs-path "../data/demo_data_test.json"