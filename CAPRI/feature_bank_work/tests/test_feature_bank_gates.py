import copy
import unittest
from feature_bank.core import FEATURES,audit_annotations

class AnnotationGates(unittest.TestCase):
    def setUp(self):
        self.manifest={'images':[{'id':'x'}]}
        self.draft={'manifest_sha256':'abc','images':[{'id':'x','labels':{f:'uncertain' for f in FEATURES}}]}
        self.audit=dict(role='user',reviewer='Test fixture only',manifest_sha256='abc',
          images=[dict(id='x',labels={f:'uncertain' for f in FEATURES},reviewed=True)])
    def test_assistant_draft_cannot_become_audit(self):
        for role in ('assistant',''):
            self.audit['role']=role
            with self.assertRaises(ValueError): audit_annotations(self.manifest,self.draft,self.audit)
    def test_unreviewed_rows_rejected(self):
        self.audit['images'][0]['reviewed']=False
        with self.assertRaises(ValueError): audit_annotations(self.manifest,self.draft,self.audit)
    def test_unknown_is_preserved_and_disagreements_recorded(self):
        self.assertEqual(audit_annotations(self.manifest,self.draft,self.audit),[])
        self.audit['images'][0]['labels']['dot_scatter']='present'
        differences=audit_annotations(self.manifest,self.draft,self.audit)
        self.assertEqual(differences,[dict(id='x',feature='dot_scatter',draft='uncertain',audited='present')])
    def test_wrong_manifest_rejected(self):
        self.audit['manifest_sha256']='wrong'
        with self.assertRaises(ValueError): audit_annotations(self.manifest,self.draft,self.audit)
    def test_missing_features_and_invalid_labels_rejected(self):
        for labels in ({},{f:'unknown' for f in FEATURES}):
            self.audit['images'][0]['labels']=labels
            with self.assertRaises(ValueError): audit_annotations(self.manifest,self.draft,self.audit)
    def test_duplicate_or_missing_ids_rejected(self):
        for images in ([],self.audit['images']*2,[dict(self.audit['images'][0],id='wrong')]):
            self.audit['images']=images
            with self.assertRaises(ValueError): audit_annotations(self.manifest,self.draft,self.audit)

if __name__=='__main__': unittest.main()
