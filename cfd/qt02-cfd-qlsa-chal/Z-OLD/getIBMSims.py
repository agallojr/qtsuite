"""
show the backends available
"""

from lwfm.midware.LwfManager import lwfManager

exec_site = lwfManager.getSite("ibm-quantum")

print(exec_site.getSpinDriver().listComputeTypes())
