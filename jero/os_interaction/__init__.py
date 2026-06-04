"""OS interaction handlers (filesystem, app launching, document creation).

Planned for a follow-up PR. All handlers will be sandboxed (filesystem root +
app allow-list) and async (blocking calls offloaded to the shared executor).
"""
