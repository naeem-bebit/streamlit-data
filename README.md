![Linter](https://github.com/naeem-bebit/streamlit-data/workflows/Linter/badge.svg)

1. Run the streamlit apps

   ```console
   $ streamlit run python_file.py
   ```

1. Set the timeout if the installation package error occured

   ```console
   $ pip install -r requirements.txt --default-timeout=200
   ```

1. Docker build and docker run

   ```console
   docker build -t streamlit_app .
   docker run streamlit_app
   ```

## Reference

1. https://www.docker.com/blog/tag/python-env-series/
