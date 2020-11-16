all: xkcd.ttf clean

FONT = Yahfie/Yahfie-Heavy.ttf
ZIPFILE = Yahfie-Normal.font.zip

xkcd.ttf: fontname.py ~/.fonts
	wget https://www.ffonts.net/$(ZIPFILE)
	unzip -p $(ZIPFILE) $(FONT) > $@
	./fontname.py xkcd $@
	cp $@ ~/.fonts/
	rm -f ~/.cache/matplotlib/fontlist-*.json
	rm -f $(ZIPFILE)

fontname.py:
	pip install fonttools
	wget https://github.com/chrissimpkins/fontname.py/raw/master/fontname.py
	chmod +x fontname.py

~/.fonts:
	mkdir $@

clean:
	pip uninstall -y fonttools
	rm -f xkcd.ttf fontname.py


depts = \
		31 34 13 42 69 38 76 75 59 \
		33 67 30 73
nonoise = \
		idf pc gc met \
		05
curfew:
	unset DISPLAY; \
	for dept in $(nonoise); do \
		./predictor.py $$dept --two-months & \
	done; \
	for dept in $(depts); do \
		./predictor.py $$dept --two-months --noise & \
	done; \
	./predictor.py met --full & \
	wait

help.fr:
	curl -sL https://github.com/ofa-/predictor/blob/master/help.fr.md \
	| sed '/<article/ s:>:\n:' \
	| sed '1,/<article/ d; /<\/article/,$$ d' \
	| sed 's:<svg.*</svg>::g' \
	> help.fr.md.html

fetch:
	./fetch.sh

upload:
	lftp -c "open $(TARGET); mput *.png"
