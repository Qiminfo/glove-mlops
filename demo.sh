python3 src/main.py load_data --output-path "./resources" && \
python3 src/main.py clean_text --data-file "./resources/demo_data_train.json" --output-path "./resources/cleansed_data.json" && \
python3 src/main.py build_cooccurrence --cleansed-cvs "./resources/cleansed_data.json" --output-path "./resources" && \
python3 src/main.py train_glove --counter "./resources/counter.pkl" --cooccurrence-matrix "./resources/cooccurrence_matrix.npz" --output-path "./resources/glove_embedder.pkl" && \
python3 src/main.py apply --glove-embedder "./resources/glove_embedder.pkl" --labeled-docs-path "./resources/demo_data_test.json" --output-path "./resources/embedded_data.json" && \
python3 src/main.py evaluate --embedded-cvs-labeled "./resources/embedded_data.json" --labeled-docs-path "./resources/demo_data_test.json"