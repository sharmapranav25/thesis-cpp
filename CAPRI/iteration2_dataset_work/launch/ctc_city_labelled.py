from pathlib import Path
# Reuse the exact preprocessing, decoding and scoring from the real-family check.
source=(Path(__file__).parent/'ctc_real_test.py').read_text()
source=source.replace("ctc_real_test_01", "ctc_city_labelled_01")
begin=source.index("crypt=Path(")
end=source.index("results=[];summaries={}",begin)
source=source[:begin]+"city=Path('E:/john/Captcha Solver/test_set/City')\ncity_rows=[(p,p.stem) for p in sorted(city.glob('*.png')) if re.fullmatch('[0-9]{5}',p.stem)]\nfamilies={'City':city_rows}\n"+source[end:]
exec(compile(source,str(Path(__file__)),'exec'))
