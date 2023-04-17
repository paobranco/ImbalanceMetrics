import unittest
from imbalance_metrics import regression_metrics as rm
import get_y

class TestRM(unittest.TestCase):

    def setUp(self):
        self.y,self.y_pred=get_y.reg()

    def test_calculate_phi(self):
        phi=[0.0, 1.0, 0.49888226603839275, 0.05445858066614002, 0.0, 0.0, 0.0, 0.0, 0.31509705701514884, 0.0, 0.0, 0.0, 0.48994099084203707, 0.0, 0.0, 0.23445933524872664, 0.0, 0.48994099084203707, 0.0, 0.019054458370812755, 0.0, 0.0, 0.0028771178571343516, 0.1291327329780504, 0.0, 0.0, 0.0, 0.0, 0.004727095806812171, 1.0, 0.48994099084203707, 0.0, 0.0, 0.0, 0.09118108941257239, 0.0, 0.03832044427322246, 0.3489981592517827, 0.004727095806812171, 0.44844225746168204, 0.14119308799446317, 0.0, 0.0, 0.0, 0.0, 0.10002552740509159, 0.024790508660356623, 0.03121928307584038, 1.0, 0.41872728008913, 0.0, 0.0, 0.09336049905751333, 0.8693041856980427, 0.0, 0.31509705701514884, 0.8227238534050285, 0.0, 0.0, 0.0, 0.6314340670969039, 0.03121928307584038, 0.16857795683963583, 0.050188425196797296, 0.00032422113788387557, 0.0, 0.0, 0.23445933524872664, 0.0, 0.050188425196797296, 0.15432632347972935, 0.0, 0.0, 0.0, 0.034156299477897605, 0.0, 0.786436613517983, 0.10025657399207298, 0.9607456461627909, 0.0, 0.0, 0.6402572556504245, 0.30633376727169226, 0.6153746655180542, 0.0, 0.06817594373338628, 0.0, 0.00147685435688878, 1.0, 0.0, 0.0, 0.370550385592348, 0.0, 0.7320888763390824, 0.0028771178571343516, 0.0, 0.0, 0.06345488185522921, 0.0, 0.1291327329780504, 0.0, 0.0, 1.0, 0.03832044427322246, 1.0, 0.0, 0.0, 0.0595128523736463, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.016452559746019977, 0.06914248670859985, 0.28776834337322144, 0.2739480393153916, 0.48994099084203707, 0.24220899826690848, 0.0, 0.9897627271260961, 0.0, 0.5524476172263427, 1.0, 0.002359971375219484, 0.08320026667341313, 0.007878459840163646, 0.5746393427959916, 0.925718814641942, 0.0, 0.1291327329780504, 0.0034442326163525694, 0.0, 0.0, 0.1843461140290312, 0.1897119142767905, 0.0, 0.0, 0.31509705701514884, 0.9714697793849044, 0.07805117806892829, 1.0, 0.0, 1.0, 0.030881705943122104, 0.0007974235758182625, 0.0, 0.0, 0.0, 0.0, 0.016452559746019977, 0.0, 0.0, 0.0, 0.35758862382381273, 0.0, 0.0, 0.44532835364132217, 0.0, 0.8012276458586418, 0.7077213600208461, 0.0007974235758182625, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.019054458370812755, 0.0, 0.1229540914533251, 0.0, 0.0, 0.1291327329780504, 0.9744433607829008, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.28205518661385076, 0.09946291339711641, 0.09946291339711641, 0.0, 0.0, 0.10514711530951412, 0.9897627271260961, 0.6659288723670778, 0.0, 0.0, 0.03037849800953219, 0.4720695614731433, 0.1291327329780504, 0.0, 0.011793726869888964, 0.0, 0.9626032767215736, 0.3319503722508787, 0.0, 0.0, 0.007878459840163646, 0.0, 0.6191154026310555, 0.0, 0.7068993214151077, 0.0, 0.02824191032874151, 0.0, 0.0, 0.0, 0.0, 0.0, 0.21919992045918962, 0.04607365564044436, 0.0, 0.5790629908125201, 0.050188425196797296, 0.35758862382381273, 0.0, 0.0070182085979284764, 0.0028771178571343516, 0.19694870741774306, 0.1861287494298653, 0.7748345661329872, 0.0, 0.0, 0.0, 0.0, 0.10228892780869597, 0.0, 0.0, 0.004727095806812171, 0.011793726869888964, 0.664145118692292, 0.0, 0.9665456067475272, 0.007878459840163646, 0.6743935026931697, 0.0, 0.03380451541469568, 0.6092841704384154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.016452559746019977, 0.03869278761365365, 0.04115257864370859, 0.0, 0.08320026667341313, 0.6307496961165205, 0.0, 0.29845854909202035, 0.0, 1.0, 0.0, 0.0, 0.715907947507188, 0.0, 0.7118223989269457, 0.0, 0.010744406348386974, 0.9635167699973576, 0.006206018916707844, 1.0, 0.0, 0.0, 0.014501100308910568, 0.0, 0.0, 0.0, 0.0, 0.0, 0.050188425196797296, 0.6743935026931697, 0.0421168147385414, 0.9774427810310118, 0.30259692581448094, 0.016202272417960295, 0.1285095505664309, 0.0, 0.0, 0.050188425196797296, 0.0, 0.2154369016006988, 0.0, 0.0, 0.9876641656787365, 0.0, 0.0, 0.0, 0.05888158025424264, 0.0, 0.0, 0.0, 0.0028771178571343516, 0.0, 0.03121928307584038, 0.0, 0.0, 0.0, 0.01343191483373477, 0.35758862382381273, 0.0, 0.0, 0.45938111183480645, 0.0, 0.0, 0.0, 0.0, 0.7479965281542191, 0.00147685435688878, 0.31509705701514884, 0.0, 0.14508676572082602, 0.0, 0.04019841196625881, 0.9399890857706474, 0.0, 0.0027696809630140947, 0.0, 0.533732796266927, 0.4720695614731433, 0.19694870741774306, 0.0, 0.0, 0.920674587904202, 0.0, 0.0, 0.0, 0.18256949627050068, 0.016202272417960295, 1.0, 0.0, 0.0, 0.0, 0.0077025799367441685, 1.0, 0.9961292261281595, 0.0, 0.7240313001087126, 0.0, 0.0, 0.0, 0.0, 0.35758862382381273, 0.9897627271260961, 0.0, 0.0, 0.0, 0.5193816734689902, 0.9774427810310118, 0.4364423692505162, 0.0, 0.0, 0.024487084974136743, 0.004365280656400221, 0.0, 0.05888158025424264, 0.7280684692306579, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9399890857706474, 1.0, 0.015462284098484776, 0.7320888763390824, 0.0, 6.656478911344434e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.26191263749639515, 0.0, 0.0034442326163525694, 0.0, 0.0, 0.0, 0.2104801471550081, 0.050188425196797296, 0.8565404834413257, 0.0, 0.0, 0.0, 0.2739480393153916, 0.786436613517983, 0.0, 0.0, 0.29742623915373717, 0.39233497630975994, 0.010744406348386974, 0.0, 0.0, 0.0, 0.0, 0.05445858066614002, 0.0, 0.24220899826690848, 0.08060836475293023, 0.11689524667038363, 0.0, 0.0, 0.439994731726997, 0.050188425196797296, 0.0, 0.07059108986213494, 0.0, 0.015219254480879907, 0.0, 0.016202272417960295, 0.0, 0.0, 0.11391154223597869, 0.0, 1.0, 1.0, 0.00016574994546618822, 0.0, 1.0, 0.00974187662248951, 0.0, 0.5790629908125201, 0.0, 0.0007407810817800994, 0.4143122109978665, 0.0, 0.0, 0.0, 0.0, 0.05888158025424264, 0.25397628795567484, 0.0, 0.0007916664949265443, 0.0, 0.0034442326163525694, 0.0, 0.0, 0.016452559746019977, 0.0, 0.0, 0.32771801769421566, 0.0, 0.0, 0.0070182085979284764, 0.0005348849140692189, 0.0, 0.14508676572082602, 0.20427733413525512, 0.18256949627050068, 0.0, 0.00016574994546618822, 0.1897119142767905, 0.10514711530951412, 0.8227238534050285, 0.027919576307306543, 0.0, 0.0, 0.0, 0.946623386388826, 0.525697830473121, 0.0, 0.9909772359396324, 0.02183462197956671, 0.0, 0.011793726869888964, 0.0007974235758182625]
        self.assertEqual(rm.calculate_phi(self.y), phi)

    def test_phi_weighted_r2(self):
        self.assertEqual(rm.phi_weighted_r2(self.y,self.y_pred), 0.10265482067638654)

    def test_phi_weighted_mse(self):
        self.assertEqual(rm.phi_weighted_mse(self.y,self.y_pred), 1.5091869655801122)

    def test_phi_weighted_mae(self):
        self.assertEqual(rm.phi_weighted_mae(self.y,self.y_pred), 0.8809891820598452)

    def test_phi_weighted_root_mse(self):
        self.assertEqual(rm.phi_weighted_root_mse(self.y,self.y_pred), 1.2284897091877132)

    def test_ser_t(self):
        self.assertEqual(rm.phi_weighted_root_mse(self.y,self.y_pred), 1.2284897091877132)

    def test_sera(self):
        self.assertEqual(rm.sera(self.y,self.y_pred), 128.2178637371627)


if __name__ == '__main__':
    unittest.main()