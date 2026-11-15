//! Fiche de la vidéo, lue uniquement dans le nom du fichier : titre, année, épisode, source…
//! Rien n'est cherché ailleurs, et rien n'est inventé : ce qui n'est pas écrit reste vide.

use std::path::Path;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ReleaseInfo {
    pub title: String,
    pub year: Option<String>,
    pub season: Option<u32>,
    pub episode: Option<u32>,
    pub episode_title: String,
    pub edition: String,
    pub source: String,
    pub service: String,
    pub resolution: String,
    pub video: String,
    pub audio: String,
    pub group: String,
}

/// Mots qu'un titre peut contenir : reconnus seulement une fois le titre terminé.
fn is_ambiguous(lower: &str) -> bool {
    matches!(lower, "web" | "dd" | "dv" | "dts" | "flac" | "mp3" | "opus" | "aac" | "hdr" | "4k" | "sdr" | "atmos")
}

fn resolution_of(lower: &str) -> Option<&'static str> {
    Some(match lower {
        "480p" => "480p",
        "576p" => "576p",
        "720p" => "720p",
        "1080p" | "1080i" => "1080p",
        "1440p" => "1440p",
        "2160p" | "4k" => "2160p",
        _ => return None,
    })
}

fn source_of(lower: &str) -> Option<&'static str> {
    Some(match lower {
        "bluray" | "blu-ray" | "bdrip" | "brrip" | "bdremux" => "Blu-ray",
        "web-dl" | "webdl" => "WEB-DL",
        "webrip" => "WEBRip",
        "web" => "WEB",
        "hdtv" => "HDTV",
        "dvdrip" | "dvd" => "DVD",
        "hdrip" => "HDRip",
        _ => return None,
    })
}

/// Sigles des plateformes, écrits en majuscules dans les noms de fichiers.
fn service_of(word: &str) -> Option<&'static str> {
    Some(match word {
        "NF" => "Netflix",
        "ATVP" => "Apple TV+",
        "AMZN" => "Amazon",
        "DSNP" => "Disney+",
        "HMAX" | "MAX" => "Max",
        "HULU" => "Hulu",
        "PCOK" => "Peacock",
        "PMTP" => "Paramount+",
        _ => return None,
    })
}

fn video_of(lower: &str) -> Option<&'static str> {
    Some(match lower {
        "x264" => "x264",
        "x265" => "x265",
        "h264" | "avc" => "H.264",
        "h265" | "hevc" => "H.265",
        "av1" => "AV1",
        "vp9" => "VP9",
        "xvid" => "XviD",
        _ => return None,
    })
}

fn video_extra_of(lower: &str) -> Option<&'static str> {
    Some(match lower {
        "10bit" | "10-bit" | "10bits" => "10 bits",
        "8bit" | "8-bit" => "8 bits",
        "hdr" | "hdr10" => "HDR",
        "hdr10+" | "hdr10plus" => "HDR10+",
        "dv" | "dovi" => "Dolby Vision",
        "sdr" => "SDR",
        _ => return None,
    })
}

fn audio_of(code: &str) -> Option<&'static str> {
    Some(match code {
        "ddp" | "dd+" | "eac3" => "Dolby Digital+",
        "dd" | "ac3" => "Dolby Digital",
        "truehd" => "Dolby TrueHD",
        "dts-hd" | "dtshd" => "DTS-HD",
        "dts" => "DTS",
        "aac" => "AAC",
        "flac" => "FLAC",
        "opus" => "Opus",
        "mp3" => "MP3",
        _ => return None,
    })
}

fn edition_of(lower: &str) -> Option<&'static str> {
    Some(match lower {
        "imax" => "IMAX",
        "extended" => "Version longue",
        "uncut" => "Non censurée",
        "unrated" => "Non classée",
        "remastered" => "Remasterisée",
        "theatrical" => "Version cinéma",
        _ => return None,
    })
}

/// Reconnaît S01E02 et retourne (saison, épisode). Tolère un suffixe : S01E01a.
fn season_episode(word: &str) -> Option<(u32, u32)> {
    let lower = word.to_lowercase();
    let (season, episode) = lower.strip_prefix('s')?.split_once('e')?;
    let digits: String = episode.chars().take_while(char::is_ascii_digit).collect();
    Some((season.parse().ok()?, digits.parse().ok()?))
}

fn current_year() -> u32 {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    1970 + (seconds / 31_556_952) as u32
}

/// Une année à venir n'en est pas une : c'est un morceau de titre (« Blade Runner 2049 »).
fn is_year(word: &str) -> bool {
    word.len() == 4 && word.parse::<u32>().is_ok_and(|year| (1900..=current_year() + 1).contains(&year))
}

/// Fin de nom qui n'est pas un groupe : le second morceau d'un mot composé.
fn is_compound_tail(after: &str) -> bool {
    matches!(after.to_lowercase().as_str(), "dl" | "hd" | "ray" | "rip" | "bit")
}

fn is_technical(word: &str) -> bool {
    let lower = word.to_lowercase();
    let code: String = lower.chars().take_while(|c| !c.is_ascii_digit() || lower.starts_with('x')).collect();
    word.chars().all(|c| c.is_ascii_digit())
        || resolution_of(&lower).is_some()
        || source_of(&lower).is_some()
        || service_of(word).is_some()
        || video_of(&lower).is_some()
        || video_extra_of(&lower).is_some()
        || audio_of(&code).is_some()
        || lower == "atmos"
        || lower == "remux"
}

/// Sépare le nom du groupe de diffusion, en fin de nom : `…x264-GROUPE`, `…-[GROUPE]`, `… [GROUPE]`.
fn split_group(stem: &str) -> (&str, String) {
    let trimmed = stem.trim_end();
    if let Some(open) = trimmed.strip_suffix(']').and_then(|s| s.rfind('[')) {
        let inner = &trimmed[open + 1..trimmed.len() - 1];
        // « [1080p] » est une information technique, pas un groupe
        if !inner.split(['.', ' ']).any(is_technical) {
            let before = trimmed[..open].trim_end().trim_end_matches('-').trim_end();
            return (before, inner.trim().to_string());
        }
    }
    if let Some((before, after)) = trimmed.rsplit_once('-') {
        let plain = !after.is_empty() && !after.contains([' ', '.']) && !is_compound_tail(after);
        let last = before.rsplit(['.', ' ', '_']).next().unwrap_or("");
        // Après un mot technique seulement : « Butt-head » reste dans le titre
        if plain && is_technical(last) {
            return (before, after.to_string());
        }
    }
    (trimmed, String::new())
}

fn push_unique(list: &mut Vec<&'static str>, value: &'static str) {
    if !list.contains(&value) {
        list.push(value);
    }
}

pub fn parse(stem: &str) -> ReleaseInfo {
    let (body, group) = split_group(stem);
    let cleaned: String = body.chars().map(|c| if matches!(c, '.' | '_' | '(' | ')' | '[' | ']') { ' ' } else { c }).collect();
    let tokens: Vec<&str> = cleaned.split_whitespace().collect();

    let mut info = ReleaseInfo { group, ..Default::default() };
    let mut title: Vec<&str> = Vec::new();
    let mut episode_title: Vec<&str> = Vec::new();
    let mut year: Option<&str> = None;
    let mut tags_started = false;
    let (mut video, mut audio, mut edition): (Vec<&str>, Vec<String>, Vec<&str>) = (Vec::new(), Vec::new(), Vec::new());
    let (mut remux, mut uhd) = (false, false);

    let mut index = 0;
    while index < tokens.len() {
        let word = tokens[index];
        let lower = word.to_lowercase();
        let next = tokens.get(index + 1).copied().unwrap_or("");
        index += 1;

        if info.season.is_none() {
            if let Some((season, episode)) = season_episode(word) {
                info.season = Some(season);
                info.episode = Some(episode);
                continue;
            }
        }
        // Une année en tête fait partie du titre (« 1917 ») : il faut un titre avant elle
        if !tags_started && is_year(word) && !title.is_empty() && info.season.is_none() {
            // Deux années de suite : la première appartient au titre
            if let Some(earlier) = year.replace(word) {
                title.push(earlier);
            }
            continue;
        }

        // Une fois le titre fini, les mots à double sens sont bien des informations techniques
        let title_over = tags_started || year.is_some() || info.season.is_some();
        let allowed = title_over || !is_ambiguous(&lower);
        if allowed {
            if let Some(resolution) = resolution_of(&lower) {
                if info.resolution.is_empty() {
                    info.resolution = resolution.to_string();
                }
                tags_started = true;
                continue;
            }
            if let Some(source) = source_of(&lower) {
                info.source = source.to_string();
                remux |= lower == "bdremux";
                tags_started = true;
                continue;
            }
            if lower == "remux" {
                remux = true;
                tags_started = true;
                continue;
            }
            if lower == "uhd" {
                uhd = true;
                tags_started = true;
                continue;
            }
            if let Some(codec) = video_of(&lower) {
                push_unique(&mut video, codec);
                tags_started = true;
                continue;
            }
            // « H.264 » et « H 264 », une fois les points retirés
            if lower == "h" && (next == "264" || next == "265") {
                push_unique(&mut video, if next == "264" { "H.264" } else { "H.265" });
                index += 1;
                tags_started = true;
                continue;
            }
            if let Some(extra) = video_extra_of(&lower) {
                push_unique(&mut video, extra);
                tags_started = true;
                continue;
            }
            if lower == "atmos" {
                audio.push("Atmos".to_string());
                tags_started = true;
                continue;
            }
            // Audio : DDP5.1, AAC2.0, TrueHD 7.1, DTS-HD MA 5.1… les points étant devenus des espaces
            let code: String = lower.chars().take_while(|c| !c.is_ascii_digit()).collect();
            let attached: String = lower.chars().skip_while(|c| !c.is_ascii_digit()).collect();
            if let Some(name) = audio_of(&code) {
                if attached.is_empty() || attached.chars().all(|c| c.is_ascii_digit()) {
                    let mut text = name.to_string();
                    if tokens.get(index).is_some_and(|t| t.eq_ignore_ascii_case("ma")) {
                        text.push_str(" MA");
                        index += 1;
                    }
                    let is_digit = |t: Option<&&str>| t.is_some_and(|t| t.len() == 1 && t.chars().all(|c| c.is_ascii_digit()));
                    if !attached.is_empty() && is_digit(tokens.get(index)) {
                        text.push_str(&format!(" {attached}.{}", tokens[index]));
                        index += 1;
                    } else if attached.is_empty() && is_digit(tokens.get(index)) && is_digit(tokens.get(index + 1)) {
                        text.push_str(&format!(" {}.{}", tokens[index], tokens[index + 1]));
                        index += 2;
                    }
                    audio.push(text);
                    tags_started = true;
                    continue;
                }
            }
        }
        if title_over {
            if let Some(service) = service_of(word) {
                info.service = service.to_string();
                tags_started = true;
                continue;
            }
            if let Some(name) = edition_of(&lower) {
                push_unique(&mut edition, name);
                continue;
            }
            if matches!(lower.as_str(), "proper" | "repack" | "rerip" | "internal") {
                tags_started = true;
                continue;
            }
        }

        // Un mot ordinaire : il compte seulement avant les informations techniques
        if tags_started {
            continue;
        }
        if info.season.is_some() {
            // « S01E02 - Door-To-Door » : le tiret n'est qu'un séparateur
            if word.chars().any(char::is_alphanumeric) {
                episode_title.push(word);
            }
        } else if year.is_none() && !word.chars().all(|c| matches!(c, '-' | '–' | '—')) {
            // Un « & » appartient au titre, un tiret seul n'est qu'un séparateur
            title.push(word);
        }
    }

    info.title = title.join(" ").trim_end_matches(['-', ' ']).to_string();
    info.year = year.map(str::to_string);
    info.episode_title = episode_title.join(" ");
    info.edition = edition.join(", ");
    if uhd && info.source == "Blu-ray" {
        info.source = "Blu-ray UHD".to_string();
    }
    if uhd && info.resolution.is_empty() {
        info.resolution = "2160p".to_string();
    }
    if remux {
        info.source = if info.source.is_empty() { "Remux".to_string() } else { format!("{} Remux", info.source) };
    }
    info.video = video.join(", ");
    info.audio = audio.join(", ");
    info
}

/// Fiche d'un fichier. Un épisode nommé seulement « S01E03 - Sign Here » prend son titre
/// dans le nom de son dossier.
pub fn describe(video: &Path) -> ReleaseInfo {
    let stem = video.file_stem().unwrap_or_default().to_string_lossy();
    let mut info = parse(&stem);
    if info.title.is_empty() {
        if let Some(folder) = video.parent().and_then(Path::file_name) {
            let from_folder = parse(&folder.to_string_lossy());
            info.title = from_folder.title;
            if info.year.is_none() {
                info.year = from_folder.year;
            }
        }
    }
    if info.title.is_empty() {
        info.title = stem.into_owned();
    }
    info
}

impl ReleaseInfo {
    /// Lignes de la fiche tirées du nom du fichier. Résolution, vidéo et audio viennent
    /// des métadonnées de la vidéo.
    pub fn rows(&self) -> Vec<(&'static str, String)> {
        let mut rows = vec![("Titre", self.title.clone())];
        if let Some(year) = &self.year {
            rows.push(("Année", year.clone()));
        }
        if !self.source.is_empty() {
            rows.push(("Source", self.source.clone()));
        }
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn film_with_all_its_tags() {
        let info = parse("Buddy.2026.2160p.4K.WEB.x265.10bit.AAC5.1-[YTS.GG - YTS.BZ]");
        assert_eq!(info.title, "Buddy");
        assert_eq!(info.year.as_deref(), Some("2026"));
        assert_eq!(info.resolution, "2160p");
        assert_eq!(info.source, "WEB");
        assert_eq!(info.video, "x265, 10 bits");
        assert_eq!(info.audio, "AAC 5.1");
        assert_eq!(info.group, "YTS.GG - YTS.BZ");
    }

    #[test]
    fn blu_ray_is_recognized_in_all_its_spellings() {
        for name in ["Film.2019.BluRay.1080p", "Film 2019 Blu-ray 1080p", "Film.2019.1080p.BDRip.x264"] {
            assert_eq!(parse(name).source, "Blu-ray", "{name}");
        }
        let info = parse("Dune.2021.2160p.UHD.BluRay.REMUX.HDR.HEVC.TrueHD.7.1.Atmos-GRP");
        assert_eq!(info.title, "Dune");
        assert_eq!(info.source, "Blu-ray UHD Remux");
        assert_eq!(info.video, "HDR, H.265");
        assert_eq!(info.audio, "Dolby TrueHD 7.1, Atmos");
        assert_eq!(info.group, "GRP");
    }

    #[test]
    fn series_episode_from_the_files_on_this_machine() {
        let info = parse("East of Eden 2026 S01E01 Brothers 1080p NF WEB-DL DDP5 1 Atmos H 264-FLUX");
        assert_eq!(info.title, "East of Eden");
        assert_eq!(info.year.as_deref(), Some("2026"));
        assert_eq!((info.season, info.episode), (Some(1), Some(1)));
        assert_eq!(info.episode_title, "Brothers");
        assert_eq!(info.resolution, "1080p");
        assert_eq!(info.service, "Netflix");
        assert_eq!(info.source, "WEB-DL");
        assert_eq!(info.audio, "Dolby Digital+ 5.1, Atmos");
        assert_eq!(info.video, "H.264");
        assert_eq!(info.group, "FLUX");

        let info = parse("Severance.S01E01.720p.ATVP.WEBRip.x264-GalaxyTV");
        assert_eq!(info.title, "Severance");
        assert_eq!(info.service, "Apple TV+");
        assert_eq!(info.source, "WEBRip");
        assert_eq!(info.group, "GalaxyTV");
    }

    #[test]
    fn a_year_can_belong_to_the_title() {
        let info = parse("1917.2019.1080p.BluRay");
        assert_eq!((info.title.as_str(), info.year.as_deref()), ("1917", Some("2019")));
        let info = parse("Blade.Runner.2049.2017.2160p");
        assert_eq!((info.title.as_str(), info.year.as_deref()), ("Blade Runner 2049", Some("2017")));
        // Une année à venir ne peut pas être celle de la vidéo
        let info = parse("Blade.Runner.2049.1080p");
        assert_eq!((info.title.as_str(), info.year), ("Blade Runner 2049", None));
    }

    #[test]
    fn ordinary_words_stay_in_the_title() {
        let info = parse("The Web of Lies 1080p");
        assert_eq!(info.title, "The Web of Lies");
        assert_eq!(info.source, "");
        assert_eq!(parse("Mon film sans tag").title, "Mon film sans tag");
        // Un tiret dans le titre n'est pas un groupe
        let info = parse("Spider-Man");
        assert_eq!((info.title.as_str(), info.group.as_str()), ("Spider-Man", ""));
        // Ni le tiret de WEB-DL
        let info = parse("Film.2020.WEB-DL");
        assert_eq!((info.source.as_str(), info.group.as_str()), ("WEB-DL", ""));
    }

    #[test]
    fn untidy_names_from_this_machine_do_not_break() {
        let info = parse("S01E01a - Blood Drive (THOBB KingTurd Splice Job)");
        assert_eq!(info.title, "");
        assert_eq!((info.season, info.episode), (Some(1), Some(1)));
        assert_eq!(info.episode_title, "Blood Drive THOBB KingTurd Splice Job");

        let info = parse("S02E03 - Home Improvement UNCUT");
        assert_eq!(info.episode_title, "Home Improvement");
        assert_eq!(info.edition, "Non censurée");

        let info = parse("Beavis & Butt-head- 1993 MTV Music Awards Clip [OVERKiLL]");
        assert_eq!(info.title, "Beavis & Butt-head");
        assert_eq!(info.year.as_deref(), Some("1993"));
        assert_eq!(info.group, "OVERKiLL");

        // Bracket technique : pas un groupe
        let info = parse("Conclave (2024) [1080p]");
        assert_eq!((info.title.as_str(), info.year.as_deref()), ("Conclave", Some("2024")));
        assert_eq!((info.resolution.as_str(), info.group.as_str()), ("1080p", ""));
    }

    #[test]
    fn episode_without_a_title_takes_it_from_its_folder() {
        let path = Path::new("D:\\Séries\\Silicon Valley (2014)\\S02E10 - Be All You Can Be.mkv");
        let info = describe(path);
        assert_eq!(info.title, "Silicon Valley");
        assert_eq!(info.year.as_deref(), Some("2014"));
        assert_eq!(info.episode_title, "Be All You Can Be");
    }

    #[test]
    fn no_information_gives_only_what_exists() {
        let rows = describe(Path::new("D:\\S01E02.mkv")).rows();
        assert!(!rows.iter().any(|r| r.0 == "Épisode"), "la fiche garde titre, année et source");
        assert!(!rows.iter().any(|r| r.0 == "Source" || r.0 == "Année"));
        let rows = describe(Path::new("Mon film.mkv")).rows();
        assert_eq!(rows, [("Titre", "Mon film".to_string())]);
    }
}
