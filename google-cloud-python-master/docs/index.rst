.. toctree::
  :maxdepth: 1
  :hidden:

  Core Libraries <core/index>
  Asset Management <asset/index>
  AutoML <automl/index>
  BigQuery <bigquery/index>
  BigQuery Data-Transfer <bigquery_datatransfer/index>
  BigQuery Storage <bigquery_storage/index>
  Bigtable <bigtable/index>
  Container <container/index>
  Container Analysis <containeranalysis/index>
  Data Catalog <datacatalog/index>
  Data Labeling <datalabeling/index>
  Data Loss Prevention <dlp/index>
  Dataproc <dataproc/index>
  Datastore <datastore/index>
  DNS <dns/index>
  Firestore <firestore/index>
  Grafeas <grafeas/index>
  IAM <iam/index>
  IoT <iot/index>
  Key Management <kms/index>
  Natural Language <language/index>
  OSLogin <oslogin/index>
  PubSub <pubsub/index>
  Memorystore <redis/index>
  Resource Manager <resource-manager/index>
  Runtime Configuration <runtimeconfig/index>
  Scheduler <scheduler/index>
  Security Center <securitycenter/index>
  Security Scanner <websecurityscanner/index>
  Spanner <spanner/index>
  Speech <speech/index>
  Stackdriver Error Reporting <error-reporting/index>
  Stackdriver Incident Response & Management <irm/index>
  Stackdriver Logging <logging/index>
  Stackdriver Monitoring <monitoring/index>
  Stackdriver Trace <trace/index>
  Storage <storage/index>
  Talent <talent/index>
  Tasks <tasks/index>
  Text-to-Speech <texttospeech/index>
  Translate <translate/index>
  Video Intelligence <videointelligence/index>
  Vision <vision/index>
  Web Risk <webrisk/index>
  Release History <releases>

Google Cloud Client Library for Python
======================================

Getting started
---------------

For more information on setting up your Python development environment,
such as installing ``pip`` and ``virtualenv`` on your system, please refer
to `Python Development Environment Setup Guide`_ for Google Cloud Platform.

.. _Python Development Environment Setup Guide: https://cloud.google.com/python/setup

Cloud Datastore
~~~~~~~~~~~~~~~

`Google Cloud Datastore`_ is a fully managed, schemaless database for storing
non-relational data.

.. _Google Cloud Datastore: https://cloud.google.com/datastore/

Install the ``google-cloud-datastore`` library using ``pip``:

.. code-block:: console

    $ pip install google-cloud-datastore

Example
^^^^^^^

.. code-block:: python

  from google.cloud import datastore

  client = datastore.Client()
  key = client.key('Person')

  entity = datastore.Entity(key=key)
  entity['name'] = 'Your name'
  entity['age'] = 25
  client.put(entity)

Cloud Storage
~~~~~~~~~~~~~

`Google Cloud Storage`_ allows you to store data on Google infrastructure.

.. _Google Cloud Storage: https://cloud.google.com/storage/

Install the ``google-cloud-storage`` library using ``pip``:

.. code-block:: console

    $ pip install google-cloud-storage

Example
^^^^^^^

.. code-block:: python

  from google.cloud import storage

  client = storage.Client()
  bucket = client.get_bucket('<your-bucket-name>')
  blob = bucket.blob('my-test-file.txt')
  blob.upload_from_string('this is test content!')

Resources
~~~~~~~~~

* `GitHub <https://github.com/GoogleCloudPlatform/google-cloud-python/>`__
* `Issues <https://github.com/GoogleCloudPlatform/google-cloud-python/issues>`__
* `Stack Overflow <http://stackoverflow.com/questions/tagged/google-cloud-python>`__
* `PyPI <https://pypi.org/project/google-cloud/>`__
