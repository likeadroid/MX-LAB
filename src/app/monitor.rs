//! Écoute : solo et mute par canal, comme sur une table de mixage.
//! Ne touche ni aux mesures ni à l'export.

use std::collections::{BTreeMap, BTreeSet};

/// Sous ce niveau, un fader coupe la piste.
pub const FADER_FLOOR_DB: f32 = -70.0;
/// Marge au-dessus de 0 dB, sur chaque piste et sur le mix.
pub const FADER_TOP_DB: f32 = 6.0;

#[derive(Clone, Default, PartialEq, Debug)]
pub struct Monitor {
    solo: BTreeSet<usize>,
    mute: BTreeSet<usize>,
    /// Gain des faders en dB ; un fader à 0 dB n'y figure pas.
    gains: BTreeMap<usize, f32>,
    /// Fader du mix, en dB.
    master_db: f32,
}

#[derive(Clone, Copy, PartialEq)]
enum Side {
    Left,
    Right,
    Both,
}

/// Place d'un canal dans le retour stéréo, et son gain (descente ITU).
fn placement(name: &str) -> (Side, f64) {
    match name {
        "L" => (Side::Left, 1.0),
        "R" => (Side::Right, 1.0),
        "Ls" | "Lb" => (Side::Left, std::f64::consts::FRAC_1_SQRT_2),
        "Rs" | "Rb" => (Side::Right, std::f64::consts::FRAC_1_SQRT_2),
        "C" | "Cs" => (Side::Both, std::f64::consts::FRAC_1_SQRT_2),
        "LFE" => (Side::Both, 0.5),
        _ => (Side::Both, 1.0),
    }
}

fn linear_gain(db: f32) -> f32 {
    if db <= FADER_FLOOR_DB {
        0.0
    } else {
        10f32.powf(db / 20.0)
    }
}

impl Monitor {
    pub fn is_solo(&self, channel: usize) -> bool {
        self.solo.contains(&channel)
    }

    pub fn is_muted(&self, channel: usize) -> bool {
        self.mute.contains(&channel)
    }

    /// Solo et mute s'excluent sur un même canal.
    pub fn toggle_solo(&mut self, channel: usize) {
        if !self.solo.remove(&channel) {
            self.solo.insert(channel);
            self.mute.remove(&channel);
        }
    }

    pub fn toggle_mute(&mut self, channel: usize) {
        if !self.mute.remove(&channel) {
            self.mute.insert(channel);
            self.solo.remove(&channel);
        }
    }

    pub fn gain_db(&self, channel: usize) -> f32 {
        self.gains.get(&channel).copied().unwrap_or(0.0)
    }

    pub fn master_db(&self) -> f32 {
        self.master_db
    }

    pub fn set_master_db(&mut self, db: f32) {
        self.master_db = ((db * 2.0).round() / 2.0).clamp(FADER_FLOOR_DB, FADER_TOP_DB);
    }

    pub fn master_gain(&self) -> f32 {
        linear_gain(self.master_db)
    }

    /// Règle un fader : de coupé à +6 dB, par demi-dB.
    pub fn set_gain_db(&mut self, channel: usize, db: f32) {
        let db = ((db * 2.0).round() / 2.0).clamp(FADER_FLOOR_DB, FADER_TOP_DB);
        if db == 0.0 {
            self.gains.remove(&channel);
        } else {
            self.gains.insert(channel, db);
        }
    }

    pub fn gains(&self) -> &BTreeMap<usize, f32> {
        &self.gains
    }

    /// Reprend les faders gardés dans le projet.
    pub fn restore_gains(&mut self, gains: &BTreeMap<usize, f32>) {
        for (&channel, &db) in gains {
            self.set_gain_db(channel, db);
        }
    }

    /// Part de chaque canal vers la gauche et vers la droite : solo, mute, faders et réduction
    /// stéréo réunis. Mêmes règles que `filter`, sans fader tout à la réduction automatique.
    pub fn matrix(&self, names: &[String]) -> Vec<(f32, f32)> {
        let active = self.is_active();
        let audible: Vec<usize> = (0..names.len()).filter(|&c| self.audible(c) && (active || names[c] != "LFE")).collect();
        let mut matrix = vec![(0.0, 0.0); names.len()];
        if let [only] = audible.as_slice() {
            let gain = self.gain(*only);
            matrix[*only] = (gain, gain);
            return matrix;
        }
        let placed: Vec<(usize, Side, f64)> = audible
            .iter()
            .map(|&c| {
                let (side, gain) = placement(&names[c]);
                (c, side, gain)
            })
            .collect();
        let sum = |wanted: Side| -> f64 { placed.iter().filter(|p| p.1 == wanted || p.1 == Side::Both).map(|p| p.2).sum() };
        let scale = 1.0 / sum(Side::Left).max(sum(Side::Right)).max(1.0);
        for &(channel, side, gain) in &placed {
            let level = (gain * scale) as f32 * self.gain(channel);
            matrix[channel] = (
                if side != Side::Right { level } else { 0.0 },
                if side != Side::Left { level } else { 0.0 },
            );
        }
        matrix
    }

    /// Une table de mixage posée une fois pour toutes : un volume par canal et par côté.
    /// Solo, mute et faders lui envoient des commandes, sans jamais relancer la chaîne audio.
    pub fn routing_graph(&self, layout: &str) -> Option<String> {
        let names: Vec<String> = crate::signals::channel_names(layout)?.iter().map(|n| n.to_string()).collect();
        if names.is_empty() {
            return None;
        }
        let count = names.len();
        let matrix = self.matrix(&names);
        let split: String = (0..count).map(|c| format!("[s{c}]")).collect();
        let mut graph = format!("aformat=channel_layouts={layout},channelsplit=channel_layout={layout}{split};");
        for (c, (left, right)) in matrix.iter().enumerate() {
            graph += &format!("[s{c}]asplit=2[a{c}][b{c}];[a{c}]volume@l{c}={left:.5}[l{c}];[b{c}]volume@r{c}={right:.5}[r{c}];");
        }
        for (side, output) in [("l", "L"), ("r", "R")] {
            let inputs: String = (0..count).map(|c| format!("[{side}{c}]")).collect();
            graph += &format!("{inputs}amix=inputs={count}:normalize=0[{output}];");
        }
        graph += "[L][R]join=inputs=2:channel_layout=stereo";
        Some(graph)
    }

    /// Gain linéaire d'un fader.
    pub fn gain(&self, channel: usize) -> f32 {
        linear_gain(self.gain_db(channel))
    }

    pub fn clear(&mut self) {
        *self = Monitor::default();
    }

    pub fn is_active(&self) -> bool {
        !self.solo.is_empty() || !self.mute.is_empty()
    }

    /// Dès qu'un canal est en solo, seuls les canaux en solo s'entendent.
    pub fn audible(&self, channel: usize) -> bool {
        if self.solo.is_empty() {
            !self.mute.contains(&channel)
        } else {
            self.solo.contains(&channel)
        }
    }

    /// Niveaux linéaires estimés du retour stéréo (gauche, droite), à partir des crêtes de chaque
    /// canal : mêmes gains que le routage, sommés comme des signaux non corrélés. Exact en stéréo.
    pub fn master_levels(&self, names: &[String], levels: &[f32]) -> (f32, f32) {
        // Sans solo ni mute, la réduction automatique du lecteur écarte le LFE
        let active = self.is_active();
        let audible: Vec<usize> = (0..names.len().min(levels.len()))
            .filter(|&c| self.audible(c) && (active || names[c] != "LFE"))
            .collect();
        if let [only] = audible.as_slice() {
            return (levels[*only], levels[*only]);
        }
        let placed: Vec<(Side, f64, f64)> = audible
            .iter()
            .map(|&c| {
                let (side, gain) = placement(&names[c]);
                (side, gain, levels[c] as f64)
            })
            .collect();
        let gains = |wanted: Side| -> f64 { placed.iter().filter(|p| p.0 == wanted || p.0 == Side::Both).map(|p| p.1).sum() };
        let scale = 1.0 / gains(Side::Left).max(gains(Side::Right)).max(1.0);
        let output = |wanted: Side| -> f32 {
            let power: f64 = placed
                .iter()
                .filter(|p| p.0 == wanted || p.0 == Side::Both)
                .map(|p| (p.1 * scale * p.2).powi(2))
                .sum();
            power.sqrt() as f32
        };
        (output(Side::Left), output(Side::Right))
    }

    /// Filtre audio pour le lecteur. `None` : tout s'entend, sortie normale.
    ///
    /// Le retour est routé à la main : la réduction stéréo automatique écarte le LFE,
    /// qu'on ne pourrait alors jamais écouter seul.
    pub fn filter(&self, names: &[String]) -> Option<String> {
        let audible: Vec<usize> = (0..names.len()).filter(|c| self.audible(*c)).collect();
        if audible.len() == names.len() {
            return None;
        }
        match audible.as_slice() {
            [] => Some("pan=stereo|c0=0*c0|c1=0*c0".to_string()),
            // Un canal seul sort sur les deux enceintes, à plein niveau
            [only] => Some(format!("pan=stereo|c0=c{only}|c1=c{only}")),
            many => {
                let placed: Vec<(usize, Side, f64)> = many
                    .iter()
                    .map(|&c| {
                        let (side, gain) = placement(&names[c]);
                        (c, side, gain)
                    })
                    .collect();
                let sum = |wanted: Side| -> f64 {
                    placed.iter().filter(|p| p.1 == wanted || p.1 == Side::Both).map(|p| p.2).sum()
                };
                // Évite la saturation quand plusieurs canaux s'additionnent
                let scale = 1.0 / sum(Side::Left).max(sum(Side::Right)).max(1.0);
                let output = |wanted: Side| -> String {
                    let terms: Vec<String> = placed
                        .iter()
                        .filter(|p| p.1 == wanted || p.1 == Side::Both)
                        .map(|p| format!("{:.4}*c{}", p.2 * scale, p.0))
                        .collect();
                    if terms.is_empty() {
                        "0*c0".to_string()
                    } else {
                        terms.join("+")
                    }
                };
                Some(format!("pan=stereo|c0={}|c1={}", output(Side::Left), output(Side::Right)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn surround() -> Vec<String> {
        ["L", "R", "C", "LFE", "Ls", "Rs"].iter().map(|n| n.to_string()).collect()
    }

    #[test]
    fn master_of_a_stereo_source_is_its_two_channels() {
        let names = ["L".to_string(), "R".to_string()];
        assert_eq!(Monitor::default().master_levels(&names, &[0.5, 0.25]), (0.5, 0.25));
    }

    #[test]
    fn master_follows_what_is_really_heard() {
        let levels = [0.5, 0.5, 0.5, 0.9, 0.5, 0.5];
        let names = surround();
        let mut monitor = Monitor::default();
        let normal = monitor.master_levels(&names, &levels);
        assert_eq!(normal.0, normal.1);
        assert!(normal.0 > 0.0 && normal.0 < 0.5, "{normal:?}");

        // Le LFE ne compte pas dans la sortie normale
        let without_lfe = monitor.master_levels(&names, &[0.5, 0.5, 0.5, 0.0, 0.5, 0.5]);
        assert_eq!(normal, without_lfe);

        // Un canal en solo sort à plein niveau des deux côtés, LFE compris
        monitor.toggle_solo(3);
        assert_eq!(monitor.master_levels(&names, &levels), (0.9, 0.9));

        monitor.clear();
        for channel in 0..6 {
            monitor.toggle_mute(channel);
        }
        assert_eq!(monitor.master_levels(&names, &levels), (0.0, 0.0));

        monitor.clear();
        monitor.toggle_solo(0); // L
        monitor.toggle_solo(4); // Ls
        let (left, right) = monitor.master_levels(&names, &levels);
        assert!(left > 0.0 && right == 0.0, "{left} {right}");
    }

    #[test]
    fn faders_do_not_touch_the_routing() {
        let names = surround();
        let mut monitor = Monitor::default();
        monitor.set_gain_db(0, -6.0);
        assert_eq!(monitor.filter(&names), None, "le fader passe par ses volumes, pas par le routage");
    }

    #[test]
    fn routing_graph_has_two_volumes_per_channel() {
        let mut monitor = Monitor::default();
        monitor.set_gain_db(0, -20.0);
        let graph = monitor.routing_graph("5.1").unwrap();
        assert_eq!(graph.matches("volume@l").count(), 6);
        assert_eq!(graph.matches("volume@r").count(), 6);
        assert_eq!(monitor.routing_graph("mono"), None);
        assert_eq!(monitor.routing_graph("inconnue"), None);
    }

    #[test]
    fn matrix_follows_solo_mute_and_faders() {
        let names = surround();
        let mut monitor = Monitor::default();
        let normal = monitor.matrix(&names);
        assert_eq!(normal[3], (0.0, 0.0), "pas de LFE dans la sortie normale");
        assert!(normal[0].0 > 0.0 && normal[0].1 == 0.0, "gauche à gauche");
        assert!(normal[1].0 == 0.0 && normal[1].1 > 0.0, "droite à droite");
        assert_eq!(normal[2].0, normal[2].1, "centre des deux côtés");
        let total: f32 = normal.iter().map(|m| m.0).sum();
        assert!(total <= 1.001, "pas de saturation : {total}");

        monitor.toggle_solo(3);
        assert_eq!(monitor.matrix(&names)[3], (1.0, 1.0), "un solo sort à plein niveau, LFE compris");
        monitor.clear();

        monitor.toggle_mute(2);
        let muted = monitor.matrix(&names);
        assert_eq!(muted[2], (0.0, 0.0));
        monitor.clear();

        monitor.set_gain_db(0, -6.0);
        let faded = monitor.matrix(&names);
        assert!((faded[0].0 / normal[0].0 - 0.501).abs() < 0.01);
        assert_eq!(faded[1], normal[1], "les autres ne remontent pas");
    }

    #[test]
    fn gains_come_back_from_the_project() {
        let mut first = Monitor::default();
        first.set_gain_db(2, -12.0);
        let mut second = Monitor::default();
        second.restore_gains(first.gains());
        assert_eq!(second.gain_db(2), -12.0);
    }

    #[test]
    fn fader_stays_in_range_and_cuts_at_the_bottom() {
        let mut monitor = Monitor::default();
        monitor.set_gain_db(1, 12.0);
        assert_eq!(monitor.gain_db(1), 6.0, "pas plus de +6 dB");
        assert!((monitor.gain(1) - 1.995).abs() < 0.01);
        monitor.set_master_db(12.0);
        assert_eq!(monitor.master_db(), 6.0);
        monitor.set_gain_db(1, -200.0);
        assert_eq!(monitor.gain(1), 0.0);
        monitor.set_gain_db(1, -3.26);
        assert_eq!(monitor.gain_db(1), -3.5);
    }

    #[test]
    fn nothing_active_leaves_the_output_alone() {
        assert_eq!(Monitor::default().filter(&surround()), None);
        assert!(!Monitor::default().is_active());
    }

    #[test]
    fn solo_wins_over_everything_else() {
        let mut monitor = Monitor::default();
        monitor.toggle_mute(0);
        monitor.toggle_solo(2);
        assert!(monitor.audible(2));
        assert!(!monitor.audible(1), "un autre canal se tait, même sans mute");
        monitor.toggle_solo(2);
        assert!(monitor.audible(1), "fin du solo : retour à la normale");
        assert!(!monitor.audible(0), "le mute reste");
    }

    #[test]
    fn solo_and_mute_exclude_each_other_on_a_channel() {
        let mut monitor = Monitor::default();
        monitor.toggle_mute(1);
        monitor.toggle_solo(1);
        assert!(monitor.is_solo(1) && !monitor.is_muted(1));
        monitor.toggle_mute(1);
        assert!(monitor.is_muted(1) && !monitor.is_solo(1));
        monitor.clear();
        assert!(!monitor.is_active());
    }

    #[test]
    fn a_single_channel_comes_out_on_both_speakers() {
        let mut monitor = Monitor::default();
        monitor.toggle_solo(3); // LFE
        assert_eq!(monitor.filter(&surround()).unwrap(), "pan=stereo|c0=c3|c1=c3");
    }

    #[test]
    fn muting_everything_gives_silence() {
        let mut monitor = Monitor::default();
        for channel in 0..6 {
            monitor.toggle_mute(channel);
        }
        assert_eq!(monitor.filter(&surround()).unwrap(), "pan=stereo|c0=0*c0|c1=0*c0");
    }

    #[test]
    fn muted_channels_disappear_from_the_mix_and_gains_cannot_clip() {
        let mut monitor = Monitor::default();
        monitor.toggle_mute(2); // C
        let filter = monitor.filter(&surround()).unwrap();
        assert!(!filter.contains("*c2"), "{filter}");
        let (left, right) = filter.strip_prefix("pan=stereo|c0=").unwrap().split_once("|c1=").unwrap();
        assert!(left.contains("*c0") && !left.contains("*c1") && left.contains("*c4"), "{left}");
        assert!(right.contains("*c1") && !right.contains("*c0") && right.contains("*c5"), "{right}");
        let sum: f64 = left.split('+').map(|t| t.split('*').next().unwrap().parse::<f64>().unwrap()).sum();
        assert!(sum <= 1.0 + 1e-3, "{sum}");
    }
}
