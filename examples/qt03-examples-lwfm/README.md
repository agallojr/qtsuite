
# README.md


Some workflow examples which focus more on lwfm features and less on the actual application logic.

lwfm provides two main software parts:
1. a Site interface, which can be extended for custom implementations masking the details of connecting to diverse compute resources.
2. a middleware component, the lwfManager, which fills in the spaces between.

Using these two components, you can create workflows that are flexible and adaptable to different environments. You'll pre-run the lwfManager locally using lwfm.sh (see that git project), then run your workflows across sites.


# Site interface

## Auth

The Auth pillar of a Site provides the means to login and check the status of authentication.

Local sites have no "login", but remote sites do. The ~/.lwfm/sites.toml defines the site and any credentials for it. A call on site.getAuthDriver().login() will use those credentials.

The lwfm site driver for the IBM quantum cloud is the best example of a real login. You can find it here:

https://github.com/lwfm-proj/ibm-quantum-site/blob/ddc7c4237074686c83eb7770294d91a77863e42a/examples/ex1_test_circuit.py

Site drivers are designed to be stateless, so it might be expected for a Site's Run driver, for example, to inquire about authentication from its own Site's Auth driver. This means its often not necessary to call site.getAuthDriver().login() before using the Run driver, as the Run driver will often do it for you if needed, depending on the site. 


## Run

The Run pillar of a Site provides the means to submit jobs, monitor their status, and retrieve results. It abstracts the complexities of job submission and execution across different compute resources.

Our prior examples of using the Run subsystem are several in this repository. Let's highlight a few.

qt01-examples-wciscc2025\test01.py shows constructing a JobDefn - the abstract definition of a job - and then iterating over a set of compute types (see Spin below) submitting the job to each. In workflow terms this is a "fan out", and lwfm tracks the parent-child relationships.  

The same example shows synchronous waiting on a job execution. This is fine for the "local" site execution, but generally not recommended for remote sites. Instead, you should use the async job submission and monitoring features of lwfm.

qt01-examples-wciscc2025\test02.py shows how to use the async job submission and monitoring features of lwfm. This example submits a job, then sets an event listener to be managed by the lwfManager. When the job reaches the target state, the event handler is triggered, and it can then take further actions, such as submitting another job or sending a notification. The example shows both cases. In workflow terms, this is a "fan in" or "join".

At any time, using the Run driver you can ask for the status of a job by its id. Often the author of the site driver will make use of lwfm to answer the question - lwfm will be holding the latest job status in store, and if its a terminal status, it can be returned to the inquiring user. Otherwise for remote sites, the Run driver can make a call out to the site to inquire. For remote jobs, the lwfManager will be doing this automatically. qt01-examples-wciscc2025\test01.py shows making a call to inquire about a job status.


## Repo 

The Repo pillar of a Site provides the means to manage and interact with data located on the site. This includes uploading and downloading files - put and get. There is no assumption that a Site provides metadata handling as most do not - that function is provided by the lwfManager.

So a call on site.getRepoDriver().put() will permit putting a local file to a site. Similarly, site.getRepoDriver().get() will retrieve a file from the site. Here "file" is a generic term as some sites (e.g. IBM quantum cloud site) has no concept of "file" per se. The implementation of the IBM Repo driver masks these details, allowing the user to perceive it in file terms - its the purpose of the Site interface to abstract these details.

Calls on put() and get() can include commentary - arbitrary metadata - provided by the workflow to indicate to future archeologists what the transaction is about. This metadata is stored in the lwfManager's store, and can be searched later using lwfManager (see examples below).

Sometimes, when working with local files (e.g. those created by a 3rd party app), the idea of a put() or get() seems inappropriate - the file is already where we want it to be. In those cases the lwfManager can be used directly to notate the file with respect to the workflow. qt01-examples-wciscc2025\test01.py shows an example of calling a notate() method for this reason. 


## Spin

Sites can offer multiple "compute types". It is possible using object oriented Python to model these types as Site classes, but if the differences are small (e.g. in name only), then this approach is simpler for the site class implementor. By calling site.getSpinDriver().listComputeTypes() you can get a list of available types. A site might offer a provisioning / deprovisioning mechanism, and maybe even billing. 

The IBM quantum cloud site driver is a good example of a site that offers multiple compute types. The IBM driver offers both real and simulator quantum compute types. This example shows usage and you can find it here:

https://github.com/lwfm-proj/ibm-quantum-site/blob/ddc7c4237074686c83eb7770294d91a77863e42a/examples/ex1_test_circuit.py


# lwfManager

The lwfManager is the middleware component that orchestrates the execution of workflows across different sites. It provides:
- a site driver factory, masking details of site interactions
- job submissions normalizing things like job ids and timestamps
- monitoring of async jobs running locally and remotely for updated status
- event handlers which trigger jobs in response to job and data events
- a centralized logging mechanism for all workflow-related activities and metadata, with search


## lwfm sites.toml

Here's a modified example of a ~/.lwfm/sites.toml file, which defines the site interfaces. It says a few things:
- the lwfManager is running on localhost:3000
- the ibm-quantum-site is a remote site with a specific token and virtual environment, and it provides its own implementations of Auth, Run, Repo, and Spin drivers
- the "local" site need not be explicitly defined, though this config file makes it easy to create custom "local" sites for specific needs.

```toml

[lwfm]
host = "127.0.0.1"
port = "3000"
emailKey = "my email server credentials here"
emailMe = "<my email address here>"

[ibm-quantum-venv]
token = "my ibm quantum token here"
venv  = "~/proj/src/ibm-quantum-site/.venv"
auth = "ibm_quantum_site.IBMQuantumSite.IBMQuantumSiteAuth"
run  = "ibm_quantum_site.IBMQuantumSite.IBMQuantumSiteRun"
repo = "ibm_quantum_site.IBMQuantumSite.IBMQuantumSiteRepo"
spin = "ibm_quantum_site.IBMQuantumSite.IBMQuantumSiteSpin"
remote = true
```


# Notes for corporate users

Corporate firewalls will harm your health. You'll sometimes have trouble downloading libraries needed for a qtsuite project. You'll have trouble connecting to remote sites. You'll have trouble with email notifications. Such is the corporate environment - to be avoided if possible.







