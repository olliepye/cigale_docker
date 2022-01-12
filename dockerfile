FROM python:3.9

Label maintainer="Ollie Pye <o.pye@hdr.qut.edu.au>"



RUN apt update && \
    apt install -y git vim && \
    pip install astropy numpy scipy db-sqlite3 sqlalchemy matplotlib configobj && \
    git clone https://gitlab.lam.fr/gyang/cigale.git && \
    git clone https://github.com/olliepye/ZFOURGE_filters.git Zfilters && \
    mv Zfilters/* cigale/database_builder/filters/ && \
    rm -fr Zfilters/ && \
    cd cigale/ && \
    grep -rl "WMAP7" | xargs sed -i 's/WMAP7/Planck18/g' && \
    python setup.py build && \
    python setup.py develop