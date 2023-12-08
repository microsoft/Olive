# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
class CompositeMixin:
    def set_composite_parent(self, cp):
        self.composite_parent = cp

    def get_composite_parent(self):
        return self.composite_parent
