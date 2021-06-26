# EMG_Tutorials

In order to run the tutorial, first get the dataset from google drive: 

Next, run the main_construct_dataset function.  This performs windowing of the dataset and stores the dataset as .npy files.

These files can be used in the subsequent tutorial scripts.

1. Within Subject Handcrafted Feature Pipeline:
This pipeline performs gesture recognition using statistical features (for instance, mean absolute value, zero crossings, slope sign change, waveform length) and statistical classifiers (linear discriminant analysis, support vector machine).
In progress

2. Within Subject Deep learning Pipeline Using Convolutional Neural Networks
In progress


3. Between Subject Handcrafted Feature Pipeline:
This shows the degradation experienced when users wish to use an sEMG gesture recognition system but do not provide data themselves through a recent acquisition protocol (See Saponas et al.).
In progress


4. Between Subject Handcrafted Features Pipeline with Projection Techniques (Canonical Correlation Analysis):
Prior to 2020, the state-of-the-art technique for achieving high performance for between subject gesture recognition relied on canonical correlation analysis (See Khushaba et al 2015).
In progress


5. Subject-Independent Adaptive Domain Adversarial Neural Network:
This technique builds a model that is well suited for many subjects.  This was a breakthrough towards between subject gesture recognition, but at this point a full acquisition protocol was still required by all end users (Cote-Allard et al 2020).
In progress


6. Between Subject Adaptive Domain Adversarial Neural Network:
This technique builds off the subject-independent adaptive domain adversarial neural network but has a mechanism to adapt to a subject that only provides a single repetition (Campbell et al 2021).
In progress




# Within Subject Results

| Subject |  TD | TDPSD | LSF4 | LSF9 |  CNN |
| --- | --- | --- | --- | --- | --- |
| S0 | 96.28059405065319 | 98.56654908559705 | 98.35276652018267 | 98.7307863078328 | 99.41452297815479 |
| S1 | 92.51591477500472 | 95.14179058412597 | 95.20592566635236 | 96.12663088881233 | 91.98451775994974 |
| S2 | 79.98360633675084 | 87.80965845505101 | 87.40853755760294 | 88.49376938758729 | 82.84690424656084 |
| S3 | 94.27212213215147 | 97.81476234326948 | 96.9501681444503 | 97.38578874842342 | 98.34340933983604 |
| S4 | 90.2387268463126 | 92.31312443007117 | 92.97773680298462 | 94.35339496396251 | 92.3039876748737 |
| S5 | 84.74000178134908 | 88.52049651157299 | 88.1928249760174 | 88.39310135008016 | 88.94057710914986 |
| S6 | 68.76059863812824 | 73.57149850658146 | 71.9219018079592 | 74.72747812834588 | 71.05533519526878 |
| S7 | 82.91067080347233 | 88.69695863046176 | 87.75371866558746 | 87.96901335158972 | 89.80586879149514 |
| S8 | 89.0038670320066 | 92.16268090014776 | 92.7841280938596 | 93.19802624276724 | 90.88853727483391 |
| S9 | 92.24744750289952 | 95.7731074458428 | 95.40857665655058 | 96.16485755785428 | 95.6602616884948 |
| Mean |  87.09535498987285 | 91.03706268927213 | 90.69562848915473 | 91.55428469272556 |  90.12439220586175 |
| STD |  7.842983256183912 | 6.870777987868703 | 7.266719880825716 | 6.734721042939276 |  7.81916163111671 |
| TIME (ms) |  0.03975517799803106 | 0.07131806242423407 | 1.3518062178069783 | 1.4118450304628045 | -- |

# Between Subject Results

## TD

| train \ test |  S0 |  S1 |  S2 |  S3 |  S4 |  S5 |  S6 |  S7 |  S8 |  S9 |  Mean |
| S0 |  NA |  98.05194805194806 | 42.89077548690876 | 43.151809292698594 | 20.682759605770602 | 47.20638540478905 | 21.256417569880206 | 19.85719385933595 | 15.434658278524449 | 25.162534828891907 | 28.094987665370027 |
| S1 |  35.48594262879977 | NA |  93.27245487622173 | 15.544928984369424 | 48.17883159548636 | 53.35661345496009 | 50.54192812321734 | 30.396287040342735 | 18.66599942808121 | 43.43787954561692 | 36.910496952686835 |
| S2 |  44.79092336235194 | 4.80131269173147 | NA |  82.88487616872457 | 17.75460648478789 | 6.021949828962372 | 12.69965772960639 | 15.158871831488755 | 16.764369459536745 | 18.91833964420947 | 17.213477939043614 |
| S3 |  30.569430569430565 | 51.58022401369765 | 13.039754478623939 | NA |  95.01499785744893 | 35.81812998859749 | 56.26069594980034 | 39.564441270974655 | 27.6880183014012 | 51.23955133242838 | 38.19620108694988 |
| S4 |  57.00727843584986 | 49.611186416494256 | 22.7535507815288 | 17.040422796743325 | NA |  91.48375142531357 | 23.7592698231603 | 22.14923241699393 | 15.885044323706033 | 27.541616060584413 | 27.796764860315754 |
| S5 |  31.53275296132439 | 68.1743597060712 | 11.869245592748555 | 62.49107270389944 | 52.12371721778791 | NA |  87.05790074158585 | 45.091038914673334 | 28.610237346296824 | 67.00007144388083 | 44.306266097878286 |
| S6 |  18.90252604538319 | 69.4513804665763 | 3.1261151952037687 | 56.599057277531784 | 45.70980615735461 | 61.986594409583574 | NA |  73.14530524812567 | 34.265084358021156 | 54.94034435950561 | 42.87504539890588 |
| S7 |  23.026973026973028 | 33.07412427766284 | 9.692384554992506 | 52.77103270961291 | 27.99315849486887 | 48.60239589275528 | 42.056408425562296 | NA |  85.63054046325422 | 31.813960134314495 | 35.82231815193039 |
| S8 |  48.865420293991725 | 71.74859099664694 | 19.27057312111912 | 45.100699900014284 | 42.524230330672744 | 56.538790644609236 | 34.45912174223491 | 16.84300829282242 | NA |  90.56226334214475 | 39.54694622996314 |
| S9 |  30.091337234194377 | 48.86209602625384 | 5.609877953036899 | 38.90158548778746 | 22.776510832383124 | 33.620935539075866 | 32.666904676901105 | 29.64684014869888 | 25.562620561548904 | NA |  29.74874538443116 |
| Mean |  29.723319435821086 | 38.0860959860633 | 38.64562271890412 | 33.37624007692886 | 27.136524113546834 | 30.444536169390215 | 37.123084462149464 | 36.39959669719822 | 39.69403216267247 | 29.882197944800403 |

##TDPSD

| train \ test |  S0 |  S1 |  S2 |  S3 |  S4 |  S5 |  S6 |  S7 |  S8 |  S9 |  Mean |
| S0 |  NA |  99.53617810760667 | 33.13833202539773 | 41.01777175076726 | 23.49664333666619 | 48.19697833523375 | 29.834569309754706 | 23.455908604069975 | 13.382899628252787 | 26.705722654854615 | 28.257277082033212 |
| S1 |  39.61038961038961 | NA |  95.82649639723194 | 19.498965098850903 | 73.32523925153549 | 55.865165336374 | 59.72618368511123 | 40.01428061406641 | 29.217901058049755 | 59.46274201614632 | 46.54357839641496 |
| S2 |  53.51791066076781 | 1.462509809517015 | NA |  90.42181143387339 | 10.65562062562491 | 18.635974914481185 | 4.171420422133486 | 8.211353088182792 | 1.8944809837003145 | 8.294634564549547 | 12.232166703614212 |
| S3 |  26.77322677322677 | 58.57173432260826 | 5.659838698165727 | NA |  98.35023568061706 | 21.38683010262258 | 51.88248716486024 | 34.75901463762942 | 42.82956820131542 | 46.75287561620347 | 36.68831786359514 |
| S4 |  61.20308263165406 | 76.87807662124563 | 17.82171151238313 | 32.78103128124554 | NA |  93.82839224629419 | 52.14632059326868 | 39.12888254194931 | 21.10380325993709 | 48.910480817317996 | 41.67024866270332 |
| S5 |  50.706436420722135 | 74.11714346864522 | 15.059596031689388 | 67.21896871875447 | 56.13597491448119 | NA |  91.25071306332002 | 53.07390217779364 | 42.193308550185876 | 66.32135457598056 | 51.556580315824085 |
| S6 |  40.730697873555016 | 72.0054219875865 | 3.8041538790949967 | 58.98443079560063 | 50.726909920182436 | 64.71762692527096 | NA |  78.90753302392002 | 46.43980554761224 | 49.22483389297707 | 47.51187178703453 |
| S7 |  21.22877122877123 | 32.98137975315688 | 6.259367639711655 | 57.2132552492501 | 27.25912200684151 | 44.63776383342841 | 38.007854337736525 | NA |  91.87160423219902 | 25.662641994713155 | 34.26709733278932 |
| S8 |  54.90937633794777 | 76.94228436898052 | 22.767825280137036 | 52.91386944722183 | 36.488027366020525 | 65.28807758128922 | 46.53338093538022 | 30.683442951100943 | NA |  93.94155890547975 | 46.52108629680502 |
| S9 |  29.42057942057942 | 58.13654847684953 | 8.857326386410676 | 48.907298957291815 | 26.403933865450398 | 41.6001140901312 | 40.2713316672617 | 44.63111238204175 | 24.640994498821176 | NA |  35.87435997164863 |
| Mean |  32.685101174338996 | 44.44253304875892 | 39.970699071660995 | 39.34283156616534 | 32.10469523257713 | 34.5374176492318 | 38.94633484364067 | 42.1545470032467 | 44.30512893098316 | 32.633295891858744 |

## LSF4

| train \ test |  S0 |  S1 |  S2 |  S3 |  S4 |  S5 |  S6 |  S7 |  S8 |  S9 |  Mean |
| S0 |  NA |  99.71457114314258 | 36.37012199472069 | 41.28184997501963 | 25.874875017854592 | 46.507981755986314 | 27.54563605248146 | 19.371652981078185 | 13.72605090077209 | 38.14388797599486 | 29.09968544460402 |
| S1 |  46.567717996289424 | NA |  95.80509381465363 | 4.653486546285062 | 64.19797171832596 | 58.715792474344354 | 50.90559041642898 | 40.79971438771867 | 29.246496997426362 | 61.984711009502035 | 44.67417314436655 |
| S2 |  48.53004138718424 | 1.6194620817578653 | NA |  90.33616444222396 | 16.547636051992573 | 11.402508551881414 | 2.167712492869367 | 8.168511245983577 | 3.2098941950243063 | 5.815531899692791 | 12.611534626416747 |
| S3 |  34.37990580847723 | 67.97460226867375 | 11.077010919991436 | NA |  97.52892443936581 | 37.39310148232612 | 60.61751283513976 | 41.977865048197074 | 38.13983414355162 | 54.625991283846545 | 43.5098323997976 |
| S4 |  54.202939917225635 | 68.14582292930014 | 12.968381985582756 | 21.554063705184973 | NA |  94.77622576966934 | 42.35596120935539 | 41.11388789717958 | 22.304832713754646 | 39.78709723512181 | 35.625524510879124 |
| S5 |  33.68060510917654 | 62.609688235713776 | 4.8890157733209625 | 61.7340379945722 | 49.18757126567845 | NA |  90.80861380490587 | 49.460906818993216 | 41.70717758078353 | 65.16396370650854 | 44.834088317835885 |
| S6 |  33.21678321678322 | 73.11835628165798 | 1.7843123260295481 | 51.73546636194829 | 53.70581527936146 | 63.92612664004563 | NA |  76.32274187790074 | 47.926794395195884 | 47.77452311209545 | 45.97748343088582 |
| S7 |  35.014985014985015 | 26.096882357137762 | 13.98900863607166 | 47.75032138265962 | 33.55188141391106 | 26.87535653166001 | 38.421992145662266 | NA |  90.7563625965113 | 26.227048653282846 | 33.41334607748021 |
| S8 |  51.855287569573285 | 71.3776128986231 | 15.252301762900577 | 54.69218683045279 | 37.09378563283923 | 60.48203080433543 | 45.233845055337376 | 26.6871604232199 | NA |  94.14160177180825 | 43.45316882398104 |
| S9 |  27.486798915370343 | 52.621816365841475 | 18.78524016843908 | 55.24925010712756 | 30.05273660205245 | 26.689960068454077 | 40.96394144948233 | 44.17357735201601 | 28.120311495320426 | NA |  36.01595916934486 |
| Mean |  32.148818398966604 | 43.152763238638585 | 40.143479850422025 | 38.19563125054392 | 30.4659095964697 | 33.92991590506485 | 35.76492035606229 | 41.92089214108315 | 41.58564113354625 | 31.906824074794486 |

## LSF9

| train \ test |  S0 |  S1 |  S2 |  S3 |  S4 |  S5 |  S6 |  S7 |  S8 |  S9 |  Mean |
| S0 |  NA |  99.76452119309262 | 33.637725618891345 | 41.6244379416173 | 20.2756749035852 | 50.8551881413911 | 26.846833998859097 | 19.62156372724027 | 13.390048613096942 | 33.528613274273056 | 28.158428075417653 |
| S1 |  51.113172541743964 | NA |  96.72540486552043 | 17.678966526300762 | 61.09841451221254 | 62.07953249714937 | 55.355105533371365 | 40.04998214923241 | 30.55476122390621 | 60.648710437950996 | 47.05593226786791 |
| S2 |  44.61253032681604 | 1.3412285082399942 | NA |  91.79216330026408 | 11.769747178974432 | 13.968072976054732 | 1.5188248716486026 | 7.197429489468048 | 2.4306548470117244 | 4.801028791883975 | 10.603216180296405 |
| S3 |  26.873126873126875 | 55.57537276164657 | 18.685318678181428 | NA |  98.01456934723612 | 26.432440136830103 | 59.697661152310324 | 38.214923241699395 | 38.80468973405776 | 41.88754733157105 | 39.165199366337816 |
| S4 |  57.57100042814328 | 61.26132553328102 | 14.609949325529941 | 20.761319811455508 | NA |  96.08751425313568 | 35.35367940673132 | 34.64476972509818 | 22.562196168144126 | 37.872401228834754 | 34.336900547447286 |
| S5 |  31.94662480376766 | 61.06870229007634 | 8.857326386410676 | 60.89130124267962 | 53.46351197263398 | NA |  91.65715915573303 | 49.82506247768654 | 41.38547326279669 | 68.97192255483317 | 45.00630682523885 |
| S6 |  28.492935635792776 | 69.10180495113077 | 10.905716936692599 | 56.96329095843451 | 45.48888255416191 | 56.26069594980034 | NA |  79.9785790789004 | 45.13869030597655 | 45.98128170322212 | 44.78504980236244 |
| S7 |  11.567004424147282 | 26.867375329956484 | 3.033330954250232 | 45.87201828310241 | 13.654503990877991 | 41.436109526525954 | 38.25776508389861 | NA |  91.53560194452388 | 29.677788097449454 | 29.20496383226757 |
| S8 |  56.10817753674896 | 70.69986445031033 | 23.25315823281707 | 51.606913298100274 | 38.05587229190422 | 58.58528237307473 | 44.88397001071046 | 24.06348298541607 | NA |  94.76316353504323 | 44.29918106545523 |
| S9 |  34.90794919366348 | 51.1521723621317 | 26.543430162015557 | 54.606484787887446 | 24.451254275940705 | 37.52139189960069 | 39.82149232416994 | 43.49442379182156 | 21.454597413731513 | NA |  37.105910690106946 |
| Mean |  31.216667573206166 | 42.86520436301945 | 40.623952595776764 | 36.39354995554014 | 29.870048401437273 | 31.27033293487077 | 36.98032109034862 | 38.2744118558436 | 40.51536733329962 | 31.711232549455715 |


